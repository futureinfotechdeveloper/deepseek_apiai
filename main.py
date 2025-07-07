from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel
import re
import requests
import json
from typing import Optional, List, Dict, Any
import os

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

def remove_repeated_lines(text: str) -> str:
    seen = set()
    result = []
    for line in text.splitlines():
        if line.strip() not in seen:
            seen.add(line.strip())
            result.append(line)
    return "\n".join(result)

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
            
            # Validate chart data
            if chart_data["labels"] and chart_data["data"] and len(chart_data["labels"]) == len(chart_data["data"]):
                # Ensure numeric data
                try:
                    chart_data["data"] = [float(x) for x in chart_data["data"]]
                    charts.append(chart_data)
                except (ValueError, TypeError):
                    continue
                
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

    external_info = get_external_data()
    external_text = format_external_data_for_ai(external_info)

    system_prompt = f"""
You are a BNI data analyst assistant. Analyze this data:
{external_text}


Response Guidelines:
1. For data visualization, provide charts in this exact format:
```chart
type: bar/pie/line
labels: ["Label1", "Label2"]
data: [value1, value2]
title: "Chart Title"
2.Special Notes:

   -P = Present, A = Absent in attendance records
   -TYFCB means "Thank You For Coming Back" (appreciation note)
   -Only use TYFCB when referring to attendance data

3.Keep responses concise and focused on the available data

4.If asked for database-specific queries, explain you can only analyze available performance metrics

5.For chart data:
    -Ensure labels and data arrays have the same length
    -Only use numeric values in data arrays
    -Keep titles descriptive but short
    
6. Keep responses in Proper Way Texts no use * or other in proper symbols 

7. Must include pie chart diagram for all the results  
    
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
        charts = extract_chart_suggestions(response)
        answer_text = re.sub(r'```chart\n.*?\n```', '', response, flags=re.DOTALL).strip()

        # Validate and clean charts
        valid_charts = []
        for chart in charts:
            if len(chart['labels']) == len(chart['data']):
                try:
                    # Ensure all data points are numeric
                    chart['data'] = [float(x) for x in chart['data']]
                    valid_charts.append(chart)
                except (ValueError, TypeError):
                    continue

        return {
            "answer": answer_text,
            "charts": valid_charts if valid_charts else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
@app.get("/health")
async def health_check():
   return {"status": "healthy"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)