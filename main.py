


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import mysql.connector
from pydantic import BaseModel
import re
import requests
import json
from typing import Optional, List, Dict, Any
import os

db_config = {
    'host': os.environ.get('DB_HOST'),
    'user': os.environ.get('DB_USER'),
    'password': os.environ.get('DB_PASSWORD'),
    'database': os.environ.get('DB_NAME')
}
# OpenRouter configuration
openrouter_config = {
    'base_url': 'https://api.deepseek.com',
    'api_key': os.environ.get('OPENROUTER_API_KEY'),
    'model': 'deepseek-chat'
}

# External API
external_api_url = 'https://bniapi.futureinfotechservices.in/BNI/bniapiecomm.php'

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_id: str
    question: str
    


def get_db_connection():
    return mysql.connector.connect(**db_config)

# def get_all_tables_schema():
#     conn = get_db_connection()
#     cursor = conn.cursor(dictionary=True)
#     try:
#         cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE()")
#         tables = [t['table_name'] for t in cursor.fetchall()]
#         schema_info = {}
#         for table in tables:
#             cursor.execute(f"DESCRIBE {table}")
#             columns = cursor.fetchall()
#             cursor.execute(f"SELECT * FROM {table} LIMIT 2")
#             sample = cursor.fetchall()
#             schema_info[table] = {"columns": columns, "sample_data": sample}
#         return schema_info
#     finally:
#         cursor.close()
#         conn.close()
def get_all_tables_schema():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        target_tables = ['bni_rosterreport']
        # target_tables = ['bni_rosterreport', 'bni_palms', 'bni_trainingmaster']
        schema_info = {}

        for table in target_tables:
            try:
                cursor.execute(f"DESCRIBE {table}")
                columns = cursor.fetchall()
                cursor.execute(f"SELECT * FROM {table} where activestatus = 'Active' and teamname = 'Team1'")
                sample = cursor.fetchall()
                schema_info[table] = {"columns": columns, "sample_data": sample}
            except Exception as e:
                schema_info[table] = {"error": str(e)}

        return schema_info
    finally:
        cursor.close()
        conn.close()

def format_schema_for_ai(schema_info):
    schema_text = ""
    for table, info in schema_info.items():
        schema_text += f"Table: {table}\nColumns:\n"
        for col in info['columns']:
            schema_text += f"- {col['Field']} ({col['Type']})\n"
        schema_text += "Sample Data:\n"
        for row in info['sample_data']:
            schema_text += f"{row}\n"
        schema_text += "=" * 40 + "\n"
    return schema_text

def get_external_data():
    try:
        response = requests.get(external_api_url)
        data = response.json()
        return data['data'] if data.get('status') else []
    except:
        return []

def format_external_data_for_ai(data):
    if not data:
        return "No external data available.\n"
    text = "External Employee Performance Metrics:\n"
    for i, row in enumerate(data, 1):
        text += f"{i}. {json.dumps(row)}\n"
    return text

def extract_sql_query(response_text: str):
    sql_match = re.search(r'```sql\n(.*?)\n```', response_text, re.DOTALL)
    if sql_match:
        return sql_match.group(1).strip()
    select_match = re.search(r'(SELECT .*?;)', response_text, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip()
    return None

def execute_sql_query(query: str):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(query)
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

def format_results_for_user(results: list, question: str) -> str:
    if not results:
        return "No results found."
    response = f"Query Results ({len(results)} rows):\n"
    for i, row in enumerate(results, 1):
        response += f"{i}. " + ", ".join(f"{k}: {v}" for k, v in row.items()) + "\n"
    return response

def remove_repeated_lines(text: str) -> str:
    seen = set()
    result = []
    for line in text.splitlines():
        if line.strip() not in seen:
            seen.add(line.strip())
            result.append(line)
    return "\n".join(result)

def detect_chart_type(results: list, question: str) -> Optional[str]:
    """Determine if results should be charted and what type"""
    if not results or len(results) < 2:
        return None
    
    # Check if we have numeric data
    has_numeric = any(isinstance(v, (int, float)) for row in results for v in row.values() if isinstance(v, (int, float)))
    
    if not has_numeric:
        return None
    
    question_lower = question.lower()
    
    # Simple detection logic
    if "count" in question_lower or "percentage" in question_lower or "proportion" in question_lower:
        return "pie"
    elif "trend" in question_lower or "over time" in question_lower or "month" in question_lower:
        return "line"
    elif "compare" in question_lower or "versus" in question_lower or "vs" in question_lower:
        return "bar"
    else:
        return "bar"  # default to bar chart

def generate_chart_data(results: list, chart_type: str, question: str) -> Dict[str, Any]:
    """Convert query results to chart data"""
    if not results:
        return {}
    
    chart_data = {
        "chart_type": chart_type,
        "title": f"Results for: {question[:50]}" + ("..." if len(question) > 50 else ""),
        "labels": [],
        "data": []
    }
    
    # Try to automatically determine labels and data
    if len(results[0]) >= 2:
        # Use first column for labels, second for data
        key_col = list(results[0].keys())[0]
        value_col = list(results[0].keys())[1]
        
        chart_data["labels"] = [str(row[key_col]) for row in results]
        try:
            chart_data["data"] = [float(row[value_col]) for row in results]
        except (ValueError, TypeError):
            # If conversion fails, try counting instead
            chart_data["data"] = [1 for _ in results]
    
    # Special handling for pie charts to ensure percentages
    if chart_type == "pie" and chart_data["data"]:
        total = sum(chart_data["data"])
        if total > 0:
            chart_data["data"] = [round((v/total)*100, 2) for v in chart_data["data"]]
    
    return chart_data

def extract_chart_suggestions(response_text: str) -> List[Dict[str, Any]]:
    """Extract chart configurations from AI response"""
    charts = []
    chart_matches = re.finditer(r'```chart\n(.*?)\n```', response_text, re.DOTALL)
    
    for match in chart_matches:
        try:
            chart_data = {
                "chart_type": "bar",  # default
                "labels": [],
                "data": [],
                "title": "Chart"
            }
            
            # Parse each line of the chart block
            for line in match.group(1).split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key in ['type', 'chart_type']:
                        chart_data["chart_type"] = value.lower()
                    elif key == 'labels':
                        # Handle both JSON arrays and comma-separated values
                        if value.startswith('['):
                            chart_data["labels"] = json.loads(value)
                        else:
                            chart_data["labels"] = [x.strip(' "\'') for x in value.split(',')]
                    elif key == 'data':
                        if value.startswith('['):
                            chart_data["data"] = json.loads(value)
                        else:
                            chart_data["data"] = [float(x.strip()) for x in value.split(',')]
                    elif key == 'title':
                        chart_data["title"] = value.strip(' "\'')
            
            # Only add if we have valid data
            if chart_data["labels"] and chart_data["data"]:
                charts.append(chart_data)
                
        except Exception as e:
            print(f"Error parsing chart suggestion: {e}")
            continue
    
    return charts

# Global memory and chain
chat_chain = None
memory = None

@app.on_event("startup")
async def startup_event():
    global chat_chain, memory

    schema_info = get_all_tables_schema()
    external_info = get_external_data()

    schema_text = format_schema_for_ai(schema_info)
    external_text = format_external_data_for_ai(external_info)

    system_prompt = f"""
You are a BNI or Business Network International data analyst with access to:

1. MySQL database with the following schema:
{schema_text}

2. External API data:
{external_text}

Response Guidelines:
1. For data visualization, you MUST provide charts in this exact format:
```chart
type: bar/pie/line
labels: ["Label1", "Label2"]
data: [value1, value2]
title: "Descriptive Title"

2.The chart block should be the ONLY place where this format appears

3.For SQL-related questions:

-Generate a SQL query wrapped in sql block

-Provide a brief explanation of the results

4.Special Notes:

-P = Present, A = Absent in bni_palms table

-TYFCB means "Thanks note" (not TYFTB)

-Only use TYFCB when referring to bni_palms data

5.Keep responses concise and avoid repetition

7.Members 30 Second Business Presentaion

8.Members Business Testimonials.



"""

    llm = ChatOpenAI(
        openai_api_base=openrouter_config['base_url'],
        openai_api_key=openrouter_config['api_key'],
        model_name=openrouter_config['model'],
        temperature=0.2
    )

    memory = ConversationBufferMemory(
        return_messages=True,
        max_token_limit=1500
    )

    chat_chain = ConversationChain(llm=llm, memory=memory, verbose=False)
    memory.chat_memory.add_ai_message(system_prompt)

@app.post("/chat")
async def chat_with_database(request: ChatRequest):
    try:
        global chat_chain, memory
        
        # Get AI response
        response = chat_chain.run(request.question)
        response = remove_repeated_lines(response)

        # Initialize response components
        sql_query = extract_sql_query(response)
        results = None
        charts = []
        answer_text = response  # Default to full response

        # Extract any explicit chart suggestions from AI
        charts = extract_chart_suggestions(response)

        # If we found charts in the response, remove the markdown from the answer text
        if charts:
            answer_text = re.sub(r'```chart\n.*?\n```', '', response, flags=re.DOTALL).strip()

        # Process SQL queries if found
        if sql_query:
            results = execute_sql_query(sql_query)
            formatted_results = format_results_for_user(results, request.question)
            
            # If we have results but no charts, try to auto-generate
            if not charts:
                chart_type = detect_chart_type(results, request.question)
                if chart_type:
                    charts.append(generate_chart_data(results, chart_type, request.question))
            
            # Combine the response text with results
            answer_text = f"{answer_text}\n\n{formatted_results}".strip()

        return {
            "answer": answer_text,
            "results": results,
            "charts": charts if charts else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        conn = get_db_connection()
        conn.close()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)
    
    
    
    
    