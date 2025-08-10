from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END 
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os 
import json
import requests
from bs4 import BeautifulSoup
import time
import re
load_dotenv()


# Configuration
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
PERSIST_DIRECTORY = r"C:\Users\Fares\Downloads\openrouter_rag"
COLLECTION_NAME = "openrouter_docs"
OPENROUTER_API_KEY = os.getenv('LLM_ROUTER')

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def scrape_openrouter_docs() -> list[Document]:
    """Scrape OpenRouter documentation from their website"""
    
    # OpenRouter documentation URLs
    doc_urls = [
        "https://openrouter.ai/docs",
        "https://openrouter.ai/docs/quick-start", 
        "https://openrouter.ai/docs/requests",
        "https://openrouter.ai/docs/responses",
        "https://openrouter.ai/docs/models",
        "https://openrouter.ai/docs/limits",
        "https://openrouter.ai/docs/errors",
        "https://openrouter.ai/docs/streaming",
        "https://openrouter.ai/docs/supported-models"
    ]
    
    documents = []
    
    for url in doc_urls:
        try:
            print(f"Scraping: {url}")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer"]):
                    script.decompose()
                
                # Extract title
                title = soup.title.string if soup.title else url.split('/')[-1]
                
                # Extract main content
                content_div = soup.find('main') or soup.find('article') or soup.find('div', class_=['content', 'documentation'])
                
                if content_div:
                    text = content_div.get_text()
                else:
                    text = soup.get_text()
                
                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                clean_text = ' '.join(chunk for chunk in chunks if chunk)
                
                if clean_text and len(clean_text) > 100:  # Only add if substantial content
                    doc = Document(
                        page_content=clean_text,
                        metadata={
                            "source": url,
                            "title": title,
                            "type": "documentation",
                            "category": url.split('/')[-1] if '/' in url else "general"
                        }
                    )
                    documents.append(doc)
                    print(f"✓ Successfully scraped: {title}")
                
            time.sleep(1)  # Be respectful with requests
            
        except Exception as e:
            print(f"✗ Failed to scrape {url}: {e}")
    
    return documents

def create_sample_openrouter_docs() -> list[Document]:
    """Create sample OpenRouter documentation when scraping isn't available"""
    
    sample_docs = [
        {
            "title": "OpenRouter API Overview",
            "content": """
OpenRouter API Documentation

OpenRouter is a unified API that provides access to multiple AI models through a single interface. 
Instead of managing multiple API keys and different interfaces, you can access models from OpenAI, 
Anthropic, Google, Meta, and other providers through one consistent API.

Key Features:
- Access to 100+ AI models
- Consistent API interface
- Pay-per-use pricing
- Model routing and fallbacks
- Real-time model availability

Base URL: https://openrouter.ai/api/v1/chat/completions
            """,
            "category": "overview"
        },
        {
            "title": "Available Models",
            "content": """
Available Models on OpenRouter

OpenRouter provides access to models from multiple providers with different capabilities and pricing:

OpenAI Models:
- openai/gpt-4o: Latest GPT-4 model, excellent for complex reasoning and analysis, higher cost
- openai/gpt-4: GPT-4 standard, best for complex tasks, reasoning, analysis, premium pricing
- openai/gpt-3.5-turbo: ChatGPT model, cost-effective for simple Q&A, summarization, general tasks

Anthropic Models:
- anthropic/claude-3-opus: Most capable Claude model, excellent for creative writing, analysis, premium cost
- anthropic/claude-3-sonnet: Balanced performance, good for most tasks, moderate pricing
- anthropic/claude-3-haiku: Fastest Claude model, best for simple tasks, most cost-effective

Google Models:
- google/gemini-pro: Google's flagship model, strong reasoning, competitive pricing
- google/palm-2: PaLM 2 by Google, good for general tasks

Meta Models:
- meta-llama/llama-2-70b: Large open-source model, good performance, cost-effective
- meta-llama/codellama-34b: Specialized for code generation, programming tasks, efficient pricing

Other Models:
- mistralai/mistral-7b: Efficient open-source model, very cost-effective
- mistralai/mixtral-8x7b: Mixture of experts model, good performance, reasonable cost

Model Selection Guidelines:
- For simple Q&A: Use gpt-3.5-turbo, claude-haiku, or mistral-7b (cost-effective)
- For code generation: Use codellama-34b, gpt-4, or claude-3-sonnet
- For complex analysis: Use gpt-4, claude-3-opus, or gemini-pro  
- For creative writing: Use claude-3-opus, gpt-4, or claude-3-sonnet
- For summarization: Use gpt-3.5-turbo, claude-haiku, or mistral-7b
- For math/reasoning: Use gpt-4, claude-3-opus, or gemini-pro

Pricing varies significantly - cheaper models for simple tasks, premium models for complex work.
            """,
            "category": "models"
        },
        {
            "title": "Model Pricing and Cost Optimization",
            "content": """
OpenRouter Model Pricing and Cost Optimization

Pricing Structure:
- Models are priced per token (input and output)
- Pricing varies dramatically between models
- Check current pricing at https://openrouter.ai/models

Cost Categories:
1. Budget Models (Under $1 per 1M tokens):
   - mistralai/mistral-7b: Very cost-effective
   - anthropic/claude-3-haiku: Fast and cheap
   - meta-llama models: Open-source efficiency

2. Mid-Range Models ($1-10 per 1M tokens):
   - openai/gpt-3.5-turbo: Good balance
   - google/gemini-pro: Competitive pricing
   - anthropic/claude-3-sonnet: Balanced performance

3. Premium Models ($10+ per 1M tokens):
   - openai/gpt-4: Highest capability
   - anthropic/claude-3-opus: Creative excellence
   - Latest flagship models

Cost Optimization Strategies:
- Use cheaper models for simple tasks (Q&A, summarization)
- Reserve expensive models for complex reasoning
- Consider model specialization (code models for programming)
- Monitor token usage and response length
- Use streaming to improve perceived performance without cost increase

Task-to-Model Cost Mapping:
- Simple Q&A: 80-90% cost savings with budget models
- Summarization: 70-85% cost savings with efficient models  
- Code generation: Use specialized models for best value
- Complex analysis: Premium models justified for quality
            """,
            "category": "pricing"
        },
        {
            "title": "Task Types and Model Capabilities",
            "content": """
AI Task Types and Optimal Model Selection

Task Classification:
1. Simple Question Answering
   - Factual questions, definitions, basic explanations
   - Best models: gpt-3.5-turbo, claude-haiku, mistral-7b
   - Characteristics: Short responses, factual accuracy

2. Complex Question Answering  
   - Multi-step reasoning, analysis, comparisons
   - Best models: gpt-4, claude-3-opus, gemini-pro
   - Characteristics: Detailed analysis, nuanced understanding

3. Code Generation
   - Writing functions, debugging, programming help
   - Best models: codellama-34b, gpt-4, claude-3-sonnet
   - Characteristics: Syntax accuracy, logic correctness

4. Summarization
   - Condensing text, extracting key points
   - Best models: gpt-3.5-turbo, claude-haiku, mistral-7b  
   - Characteristics: Conciseness, key information retention

5. Creative Writing
   - Stories, poems, creative content
   - Best models: claude-3-opus, gpt-4, claude-3-sonnet
   - Characteristics: Creativity, narrative flow, style

6. Analysis and Research
   - Data analysis, trend identification, research
   - Best models: gpt-4, claude-3-opus, gemini-pro
   - Characteristics: Deep thinking, pattern recognition

7. Translation
   - Language conversion, localization
   - Best models: gpt-4, claude-3-sonnet, gemini-pro
   - Characteristics: Cultural awareness, accuracy

8. Mathematical Tasks
   - Calculations, problem solving, equations
   - Best models: gpt-4, gemini-pro, claude-3-opus
   - Characteristics: Logical reasoning, accuracy

Model Specializations:
- Code: codellama, gpt-4, claude-sonnet
- Speed: claude-haiku, gpt-3.5-turbo, mistral-7b
- Reasoning: gpt-4, claude-opus, gemini-pro
- Cost: mistral-7b, claude-haiku, llama models
            """,
            "category": "tasks"
        }
    ]
    
    documents = []
    for doc_data in sample_docs:
        doc = Document(
            page_content=doc_data["content"],
            metadata={
                "source": "sample_documentation",
                "title": doc_data["title"],
                "type": "documentation", 
                "category": doc_data["category"]
            }
        )
        documents.append(doc)
    
    return documents

# Load OpenRouter documentation
try:
    print("Loading OpenRouter documentation...")
    try:
        documents = scrape_openrouter_docs()
        if not documents:
            raise Exception("No documents scraped")
        print(f"Successfully scraped {len(documents)} documents")
    except:
        print("Scraping failed, using sample documentation...")
        documents = create_sample_openrouter_docs()
        print(f"Created {len(documents)} sample documents")
        
except Exception as e:
    print(f"Error loading documents: {e}")
    raise

# Process documents
try:
    print("Processing documents...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_chunks = text_splitter.split_documents(documents)
    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    
except Exception as e:
    print(f"Error processing documents: {e}")
    raise

# Setup ChromaDB vector store
try:
    print("Creating vector store...")
    if not os.path.exists(PERSIST_DIRECTORY):
        os.makedirs(PERSIST_DIRECTORY)
    
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME
    )
    print("Vector store created successfully!")
except Exception as e:
    print(f"Error creating vector store: {e}")
    raise

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 7}
)

@tool
def openrouter_docs_tool(query: str) -> str:
    '''Search OpenRouter API documentation for information about models, authentication, requests, responses, errors, and usage'''
    try:
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in OpenRouter documentation."
        
        result_parts = []
        for i, doc in enumerate(docs):
            category = doc.metadata.get('category', 'general')
            title = doc.metadata.get('title', f'Document {i+1}')
            content = doc.page_content.strip()
            
            result_parts.append(f"[{category.upper()}] {title}:\n{content}")
        
        return "\n\n---\n\n".join(result_parts)
        
    except Exception as e:
        return f"Documentation search error: {str(e)}"

@tool 
def classify_task_tool(prompt: str) -> str:
    '''Classify the type of task/prompt to determine the best model approach'''
    
    # Query RAG for task classification guidance
    classification_query = f"classify this task type: {prompt[:200]}... What type of AI task is this - simple QA, complex analysis, code generation, summarization, creative writing, translation, math, or other?"
    
    try:
        docs = retriever.invoke(classification_query)
        context = "\n".join([doc.page_content for doc in docs[:3]])
        
        return f"Task classification context from documentation:\n{context}\n\nPrompt to classify: {prompt}"
        
    except Exception as e:
        return f"Classification error: {str(e)}"

@tool
def recommend_model_tool(task_type: str, priority: str = "cost") -> str:
    '''Get model recommendation based on task type and priority (cost, speed, quality)'''
    
    # Query RAG for model recommendations
    if priority == "cost":
        query = f"cheapest most cost effective models for {task_type} tasks budget models"
    elif priority == "speed":
        query = f"fastest quickest models for {task_type} tasks fast models"
    elif priority == "quality":
        query = f"best highest quality models for {task_type} tasks premium models"
    else:
        query = f"recommended models for {task_type} tasks"
    
    try:
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs[:4]])
        
        return f"Model recommendations for {task_type} (priority: {priority}):\n{context}"
        
    except Exception as e:
        return f"Model recommendation error: {str(e)}"

@tool
def get_model_pricing_tool(model_name: str) -> str:
    '''Get pricing and cost information for a specific model'''
    
    query = f"pricing cost rate token price for {model_name} model"
    
    try:
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs[:3]])
        
        return f"Pricing information for {model_name}:\n{context}"
        
    except Exception as e:
        return f"Pricing lookup error: {str(e)}"

@tool
def make_openrouter_request_tool(prompt: str, model: str, max_tokens: int = 1000) -> str:
    '''Make an actual API request to OpenRouter with the specified model'''
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions", 
            headers=headers, 
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return f"✅ SUCCESS using {model}:\n\n{result['choices'][0]['message']['content']}\n\nTokens used: {result.get('usage', {})}"
        else:
            return f"❌ ERROR {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"❌ Request failed: {str(e)}"

tools = [openrouter_docs_tool, classify_task_tool, recommend_model_tool, get_model_pricing_tool, make_openrouter_request_tool]

# Initialize LLM
llm = None
llm_name = "None"

# State definition
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Enhanced system prompt for routing
system_prompt = """You are an expert OpenRouter model routing assistant. Your job is to:

1. CLASSIFY the user's prompt/task type using the classify_task_tool
2. RECOMMEND the best model using recommend_model_tool based on their priority (cost/speed/quality)  
3. GET PRICING info using get_model_pricing_tool if needed
4. MAKE THE REQUEST using make_openrouter_request_tool with the recommended model
5. EXPLAIN your routing decision

Always follow this workflow:
- First classify the task type
- Then get model recommendations based on user's priority (default to cost-effective)
- Optionally check pricing if relevant
- Make the actual API request
- Explain why you chose that model
- You should balance choosing the best llm/model with the most cost effective llm . Priority is leaning towards most cost effective

Use the openrouter_docs_tool for any general OpenRouter questions.

When users ask about routing or model selection, always use the RAG system to make informed decisions rather than hardcoded rules."""

def should_continue(state: AgentState):
    """Determine if we should call tools or end"""
    last_msg = state['messages'][-1]
    return hasattr(last_msg, 'tool_calls') and bool(last_msg.tool_calls)

def call_llm(state: AgentState) -> AgentState:
    """Generate response or tool calls"""
    try:
        messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
        
        if llm is None:
            # Enhanced direct routing fallback when no LLM available
            query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
            
            print(f"🤖 Processing query: {query}")
            
            # Step 1: Classify the task using direct RAG query
            classification_query = f"classify this task type: {query[:200]}... What type of AI task is this - simple QA, complex analysis, code generation, summarization, creative writing, translation, math, or other?"
            classification_docs = retriever.invoke(classification_query)
            classification_result = "\n".join([doc.page_content for doc in classification_docs[:2]])
            print(f"📝 Task classification: Found {len(classification_docs)} relevant docs")
            
            # Step 2: Determine priority from query
            priority = "cost"  # default
            if any(word in query.lower() for word in ["fast", "quick", "speed"]):
                priority = "speed"
            elif any(word in query.lower() for word in ["best", "quality", "premium", "good"]):
                priority = "quality"
            elif any(word in query.lower() for word in ["free", "cheap", "cost", "budget"]):
                priority = "cost"
            
            # Step 3: Extract task type from query (improved classification)
            task_type = "general"
            query_lower = query.lower()
            
            if any(word in query_lower for word in ["code", "programming", "function", "script", "python", "javascript", "coding"]):
                task_type = "code generation"
            elif any(word in query_lower for word in ["story", "stories", "creative", "fiction", "poem", "narrative", "writing", "character"]):
                task_type = "creative writing"
            elif any(word in query_lower for word in ["summary", "summarize", "brief", "condense", "tldr"]):
                task_type = "summarization"
            elif any(word in query_lower for word in ["what is", "define", "explain", "simple", "quick question"]):
                task_type = "simple question answering"
            elif any(word in query_lower for word in ["analysis", "analyze", "research", "complex", "detailed", "examine"]):
                task_type = "complex analysis"
            elif any(word in query_lower for word in ["translate", "translation", "language"]):
                task_type = "translation"
            elif any(word in query_lower for word in ["math", "calculate", "equation", "solve"]):
                task_type = "math"
            
            print(f"🎯 Classified as: {task_type}")
            
            # Step 4: Smart model selection based on task type and priority
            if task_type == "creative writing":
                if priority == "cost":
                    recommended_model = "anthropic/claude-3-haiku"  # Good for creative, affordable
                elif priority == "quality":
                    recommended_model = "anthropic/claude-3-opus"   # Best for creative writing
                else:
                    recommended_model = "anthropic/claude-3-sonnet" # Balanced
                    
            elif task_type == "code generation":
                if priority == "cost":
                    recommended_model = "mistralai/mistral-7b-instruct"  # Very cost-effective for code
                elif priority == "quality":
                    recommended_model = "openai/gpt-4"                   # Best for complex code
                else:
                    recommended_model = "meta-llama/codellama-34b-instruct"  # Specialized for code
                    
            elif task_type == "simple question answering":
                if priority == "cost":
                    recommended_model = "mistralai/mistral-7b-instruct"  # Very cheap
                elif priority == "speed":
                    recommended_model = "anthropic/claude-3-haiku"       # Very fast
                else:
                    recommended_model = "openai/gpt-3.5-turbo"           # Reliable default
                    
            elif task_type == "complex analysis":
                if priority == "cost":
                    recommended_model = "google/gemini-pro"              # Good value for analysis
                elif priority == "quality":
                    recommended_model = "openai/gpt-4"                   # Best reasoning
                else:
                    recommended_model = "anthropic/claude-3-sonnet"      # Balanced
                    
            elif task_type == "summarization":
                if priority == "cost":
                    recommended_model = "mistralai/mistral-7b-instruct"  # Very efficient
                elif priority == "speed":
                    recommended_model = "anthropic/claude-3-haiku"       # Fast
                else:
                    recommended_model = "openai/gpt-3.5-turbo"           # Good default
                    
            else:  # general
                if priority == "cost":
                    recommended_model = "mistralai/mistral-7b-instruct"  # Cheapest option
                elif priority == "speed":
                    recommended_model = "anthropic/claude-3-haiku"       # Fastest
                elif priority == "quality":
                    recommended_model = "openai/gpt-4"                   # Best quality
                else:
                    recommended_model = "openai/gpt-3.5-turbo"           # Balanced default
            
            # Now get RAG verification of this choice
            verification_query = f"is {recommended_model} good for {task_type} tasks? {recommended_model} capabilities performance"
            verification_docs = retriever.invoke(verification_query)
            model_recommendation = "\n".join([doc.page_content for doc in verification_docs[:2]])
            
            # If RAG suggests a different model, try to extract it
            alternative_patterns = [
                r"(openai/gpt-[^\s,)]+)",
                r"(anthropic/claude-[^\s,)]+)", 
                r"(meta-llama/[^\s,)]+)",
                r"(mistralai/[^\s,)]+)",
                r"(google/[^\s,)]+)"
            ]
            
            rag_suggested_model = None
            for pattern in alternative_patterns:
                matches = re.findall(pattern, model_recommendation, re.IGNORECASE)
                if matches:
                    # Filter out the model we already chose
                    for match in matches:
                        if match.lower() != recommended_model.lower():
                            rag_suggested_model = match
                            break
                    if rag_suggested_model:
                        break
            
            # If RAG found a different good model, consider switching
            if rag_suggested_model and any(keyword in model_recommendation.lower() for keyword in ["better", "best", "recommended", "good for"]):
                print(f"💡 RAG suggests alternative: {rag_suggested_model}")
                recommended_model = rag_suggested_model
            
            # Step 6: Get pricing info using direct RAG query
            pricing_query = f"pricing cost rate token price for {recommended_model} model"
            pricing_docs = retriever.invoke(pricing_query)
            pricing_info = "\n".join([doc.page_content for doc in pricing_docs[:2]])
            
            # Step 7: Make the actual request
            print(f"🚀 Making API request with {recommended_model}")
            
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": recommended_model,
                "messages": [{"role": "user", "content": query}],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions", 
                    headers=headers, 
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    api_response = f"✅ SUCCESS using {recommended_model}:\n\n{result['choices'][0]['message']['content']}\n\nTokens used: {result.get('usage', {})}"
                else:
                    api_response = f"❌ ERROR {response.status_code}: {response.text}"
                    
            except Exception as e:
                api_response = f"❌ Request failed: {str(e)}"
            
            # Format comprehensive response
            response_text = f"""🎯 **DYNAMIC ROUTING ANALYSIS**

**Your Query:** {query}

**Task Classification:** {task_type} 
**Priority:** {priority} (detected from your query)
**Recommended Model:** {recommended_model}

**Why this model?** 
Based on RAG analysis of OpenRouter docs, this model is recommended for {task_type} with {priority} priority.

**Found in documentation:**
{model_recommendation[:400]}...

**Pricing Context:**
{pricing_info[:300]}...

**API Response:**
{api_response}

---
✅ **Routing complete!** Used RAG system to analyze your request and route to the most suitable model.
"""
            
            return {'messages': [AIMessage(content=response_text)]}
        
        response = llm.invoke(messages)
        return {'messages': [response]}
        
    except Exception as e:
        print(f"LLM Error: {e}")
        error_response = f"""❌ **Error in routing system**: {str(e)}

Don't worry! Let me help you manually:

For **coding tasks with free/cheap models**, here are good options:
- `mistralai/mistral-7b-instruct` - Very cost-effective for coding
- `meta-llama/llama-2-70b-chat` - Open source, good for code
- `anthropic/claude-3-haiku` - Fast and affordable
- `openai/gpt-3.5-turbo` - Reliable and reasonably priced

Would you like me to make a request with one of these models?"""
        
        return {'messages': [AIMessage(content=error_response)]}

def take_action(state: AgentState) -> AgentState:
    """Execute tool calls"""
    try:
        tool_calls = state['messages'][-1].tool_calls
        results = []
        
        for t in tool_calls:
            if t['name'] == "openrouter_docs_tool":
                result = openrouter_docs_tool.invoke(t['args']['query'])
            elif t['name'] == "classify_task_tool":
                result = classify_task_tool.invoke(t['args']['prompt'])
            elif t['name'] == "recommend_model_tool":
                args = t['args']
                result = recommend_model_tool.invoke(args['task_type'], args.get('priority', 'cost'))
            elif t['name'] == "get_model_pricing_tool":
                result = get_model_pricing_tool.invoke(t['args']['model_name'])
            elif t['name'] == "make_openrouter_request_tool":
                args = t['args']
                result = make_openrouter_request_tool.invoke(
                    args['prompt'], 
                    args['model'], 
                    args.get('max_tokens', 1000)
                )
            else:
                result = "Unknown tool"
            
            results.append(ToolMessage(
                tool_call_id=t['id'],
                name=t['name'],
                content=str(result)
            ))
        
        return {'messages': results}
    except Exception as e:
        print(f"Tool Error: {e}")
        return {'messages': [AIMessage(content="Tool execution failed.")]}

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("action", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "action", False: END}
)
graph.add_edge("action", "llm")
graph.set_entry_point("llm")

routing_agent = graph.compile()

def run_router():
    print(f"\n=== DYNAMIC OPENROUTER MODEL ROUTER (using {llm_name}) ===")
    print("I'll classify your prompt and route it to the best model based on cost-effectiveness!")
    print("\nExamples:")
    print("- 'What is machine learning?' (I'll use a cheap model)")
    print("- 'Write a complex data analysis in Python' (I'll use a capable model)")
    print("- 'Summarize this article...' (I'll use an efficient model)")
    print("- 'Create a creative story about...' (I'll use a creative model)")
    print("\nYou can also specify priority: 'Use the FASTEST model to answer: What is AI?'")
    print("Or: 'Use the BEST QUALITY model to write me a story'")
    print("\nType 'exit' to quit\n")
    
    while True:
        try:
            query = input("Your prompt: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            if not query:
                continue
                
            result = routing_agent.invoke({"messages": [HumanMessage(content=query)]})
            print("\n" + "="*80)
            print("ROUTING RESULT:")
            print("="*80)
            print(result['messages'][-1].content)
            print("="*80 + "\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")



if __name__ == "__main__":
    print(" Run interactive router")
    choice = input("Choose (Y or N): ").strip()
    
    if choice == "Y":
        run_router()
    else:
        exit 
