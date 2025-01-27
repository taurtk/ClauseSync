import streamlit as st
import requests
import asyncio
import nest_asyncio
import pdfplumber  # For PDF text extraction
import json  # For parsing JSON response

# Apply nest_asyncio for Jupyter/IPython compatibility
nest_asyncio.apply()

# Function to split text into chunks based on token limit
def split_text_into_chunks(text, max_tokens=4000):
    """
    Splits the text into chunks of approximately `max_tokens` tokens.
    Assumes 1 token ≈ 4 characters or 0.75 words.
    """
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        # Approximate token count (1 token ~= 4 characters or 0.75 words)
        if len(" ".join(current_chunk)) > max_tokens * 0.75:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Function to generate chat completion using OpenAI API
async def generate_chat_completion(api_key, system_prompt, user_prompt):
    base_url = "https://api.openai.com/v1"
    
    try:
        # Define the payload for the API request
        payload = {
            "model": "gpt-3.5-turbo",  # Use GPT-3.5 Turbo
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 1000,  # Increase tokens for detailed analysis
            "temperature": 0.5  # Lower temperature for more focused responses
        }
        
        # Send the request to the API
        response = await asyncio.to_thread(
            requests.post,
            f"{base_url}/chat/completions",
            json=payload,
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
        )
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        st.error(f'HTTP Error: {e.response.status_code} - {e.response.text}')
    except requests.exceptions.RequestException as e:
        st.error(f'API Request Error: {e}')
    return None

# Function to extract text from PDFs
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""  # Handle pages with no text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text

# Function to merge multiple JSON responses into a single JSON object
def merge_json_responses(json_responses):
    merged_result = {
        "risk_analysis": {
            "high_risk_clauses": [],
            "medium_risk_clauses": [],
            "low_risk_clauses": []
        },
        "compliance": {
            "gdpr": "Compliant",
            "data_protection": "Compliant",
            "intellectual_property": "Compliant"
        },
        "key_clauses": []
    }

    for response in json_responses:
        try:
            data = json.loads(response)
            
            # Merge risk analysis
            if "risk_analysis" in data:
                for risk_level in ["high_risk_clauses", "medium_risk_clauses", "low_risk_clauses"]:
                    if risk_level in data["risk_analysis"]:
                        merged_result["risk_analysis"][risk_level].extend(data["risk_analysis"][risk_level])
            
            # Merge compliance (take the strictest compliance)
            if "compliance" in data:
                for compliance_key in ["gdpr", "data_protection", "intellectual_property"]:
                    if compliance_key in data["compliance"]:
                        if data["compliance"][compliance_key] == "Non-compliant":
                            merged_result["compliance"][compliance_key] = "Non-compliant"
            
            # Merge key clauses
            if "key_clauses" in data:
                merged_result["key_clauses"].extend(data["key_clauses"])
        
        except json.JSONDecodeError:
            st.error(f"Failed to parse JSON response: {response}")
    
    return merged_result

# Function to analyze the contract using the LLM
async def analyze_contract(api_key, contract_text):
    # Define comprehensive system prompt
    system_prompt = """
    You are an AI-powered contract review assistant. Your task is to analyze contracts for the following aspects:
    1. Clause extraction: Identify and extract key clauses.
    2. Risk assessment: Evaluate the risk level of each clause.
    3. Anomaly detection: Detect any unusual or non-standard clauses.
    4. Compliance checking: Ensure the contract complies with relevant regulations (e.g., GDPR).
    5. Provide a detailed analysis report in the following JSON format:
    {
        "risk_analysis": {
            "high_risk_clauses": [],
            "medium_risk_clauses": [],
            "low_risk_clauses": []
        },
        "compliance": {
            "gdpr": "Compliant/Non-compliant",
            "data_protection": "Compliant/Non-compliant",
            "intellectual_property": "Compliant/Non-compliant"
        },
        "key_clauses": [
            {
                "clause_name": "Termination Clause",
                "description": "30 days' notice"
            },
            {
                "clause_name": "Liability Limitation",
                "description": "Limited to contract value"
            },
            {
                "clause_name": "Confidentiality Agreement",
                "description": "Standard clause"
            }
        ]
    }
    """
    
    # Split the contract text into smaller chunks
    chunks = split_text_into_chunks(contract_text)
    analysis_results = []

    for chunk in chunks:
        user_prompt = f"""Analyze the following contract text and provide a detailed report in JSON format:
        {chunk}
        """
        
        # Generate analysis using the LLM
        analysis_result = await generate_chat_completion(api_key, system_prompt, user_prompt)
        if analysis_result:
            analysis_results.append(analysis_result)
    
    # Combine results from all chunks into a single JSON object
    return merge_json_responses(analysis_results)

# Function to parse and display the analysis result
def display_analysis_result(analysis_result):
    try:
        # Display Risk Analysis
        st.subheader("Risk Analysis")
        st.write("**High Risk Clauses:**")
        for clause in analysis_result["risk_analysis"]["high_risk_clauses"]:
            if isinstance(clause, dict):
                st.write(f"- {clause['clause_name']}: {clause['description']}")
            else:
                st.write(f"- {clause}")
        
        st.write("**Medium Risk Clauses:**")
        for clause in analysis_result["risk_analysis"]["medium_risk_clauses"]:
            if isinstance(clause, dict):
                st.write(f"- {clause['clause_name']}: {clause['description']}")
            else:
                st.write(f"- {clause}")
        
        st.write("**Low Risk Clauses:**")
        for clause in analysis_result["risk_analysis"]["low_risk_clauses"]:
            if isinstance(clause, dict):
                st.write(f"- {clause['clause_name']}: {clause['description']}")
            else:
                st.write(f"- {clause}")
        
        # Display Compliance
        st.subheader("Compliance")
        st.write(f"**GDPR:** {analysis_result['compliance']['gdpr']}")
        st.write(f"**Data Protection:** {analysis_result['compliance']['data_protection']}")
        st.write(f"**Intellectual Property:** {analysis_result['compliance']['intellectual_property']}")
        
        # Display Key Clauses
        st.subheader("Key Clauses")
        for clause in analysis_result["key_clauses"]:
            st.write(f"**{clause['clause_name']}:** {clause['description']}")
    
    except KeyError as e:
        st.error(f"Missing expected key in analysis result: {e}")

# Streamlit UI
st.title("ClauseSync")

# Display key metrics (can be dynamically updated based on backend data)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Contracts Reviewed", "1,234")  # Replace with dynamic data
col2.metric("High Risk Contracts", "56")    # Replace with dynamic data
col3.metric("Approved Contracts", "987")    # Replace with dynamic data
col4.metric("Active Users", "42")           # Replace with dynamic data

# Upload Contract Section
st.header("Upload New Contract")
st.markdown("**Please upload a contract file (PDF, DOC, TXT) that is less than 100 KB.**")
uploaded_file = st.file_uploader("Drag and drop your contract file or click to browse", type=["pdf", "doc", "txt"])

if uploaded_file is not None:
    # Check file size
    if uploaded_file.size > 200 * 1024:  # 100 KB in bytes
        st.error("File size exceeds 200 KB. Please upload a smaller file.")
    else:
        try:
            # Extract text from the uploaded file
            if uploaded_file.type == "application/pdf":
                contract_text = extract_text_from_pdf(uploaded_file)
            else:
                contract_text = uploaded_file.read().decode("utf-8")
            
            # Analyze the contract
            if st.button("Start AI Review"):
                with st.spinner("Analyzing contract..."):
                    # Get the API key from secrets.toml
                    if "api" in st.secrets and "key" in st.secrets["api"]:
                        api_key = st.secrets["api"]["key"]
                    else:
                        st.error("API key not found in secrets.toml. Please add it.")
                        st.stop()
                    
                    # Analyze the contract using the LLM
                    analysis_result = asyncio.run(analyze_contract(api_key, contract_text))
                    
                    if analysis_result:
                        st.markdown("### Analysis Result")
                        display_analysis_result(analysis_result)  # Display the parsed analysis result
                    else:
                        st.error("Failed to analyze the contract.")
        except Exception as e:
            st.error(f"Error processing the file: {e}")

# Recent Contract Activity (can be dynamically updated based on backend data)
st.header("Recent Contract Activity")
st.write("Latest updates on contract reviews and approvals")

# Example dynamic data (replace with actual data from backend)
recent_activity = [
    {"Contract Name": "Service Agreement - TechCorp", "Status": "Approved", "Risk Level": "Low", "Last Updated": "2023-09-15"},
    {"Contract Name": "NDA - StartupX", "Status": "In Review", "Risk Level": "Medium", "Last Updated": "2023-09-14"},
    {"Contract Name": "Licensing Agreement - BigCo", "Status": "Needs Attention", "Risk Level": "High", "Last Updated": "2023-09-13"},
]

st.table(recent_activity)
