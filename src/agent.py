"""ZentroMart Conversational Support Agent powered by Google Gemini."""
import os
import pandas as pd
from datetime import datetime, timedelta
from google import genai
from google.genai import types

try:
    from tools.tools import lookup_order, create_ticket
except ImportError:
    lookup_order = None
    create_ticket = None

# OPTIMIZATION: Load CSVs into RAM once on startup instead of every tool call
try:
    PRODUCTS_DF = pd.read_csv("data/products.csv", dtype=str)
    HOLIDAYS_DF = pd.read_csv("data/holidays.csv")
    HOLIDAY_DATES = pd.to_datetime(HOLIDAYS_DF['date']).dt.date.values
except Exception as e:
    print(f"Warning: Could not load CSVs into memory: {e}")
    PRODUCTS_DF, HOLIDAY_DATES = None, []


# --- NEW STRUCTURED DATA TOOLS ---

def lookup_product(sku: str) -> dict:
    """Look up product details like warranty, return category, and assured status from the catalog."""
    if PRODUCTS_DF is None:
        return {"error": "Product database offline."}
    
    product = PRODUCTS_DF[PRODUCTS_DF['sku'] == sku]
    if not product.empty:
        return product.iloc[0].to_dict()
    return {"error": f"Product with SKU {sku} not found."}

def calculate_deadline(start_date: str, days_to_add: int, day_type: str = "business") -> str:
    """Calculate a future date. Business days exclude Sundays and dates in holidays.csv."""
    try:
        current = datetime.strptime(start_date, "%Y-%m-%d")
        if day_type == "calendar":
            return (current + timedelta(days=days_to_add)).strftime("%Y-%m-%d")
        
        added = 0
        while added < days_to_add:
            current += timedelta(days=1)
            # Skip Sundays (6) and holidays
            if current.weekday() != 6 and current.date() not in HOLIDAY_DATES:
                added += 1
        return current.strftime("%Y-%m-%d")
    except Exception as e:
        return {"error": str(e)}


# --- AGENT CLASS ---

class ZentroMartAgent:
    def __init__(self, retriever, model_name: str = "gemini-3.5-flash-lite"):
        self.retriever = retriever
        self.model_name = model_name
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("CRITICAL: GEMINI_API_KEY is blank.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.reset_history()

    def reset_history(self):
        self.system_prompt = (
            "You are an AI support assistant for ZentroMart, an Indian online store. "
            "Today's reference date is Tuesday, 2026-10-20. "
            "Rules:\n"
            "1. Answer ONLY using the provided Context Documents and tool outputs.\n"
            "2. To verify a user for order details, use `lookup_order` with their Order ID and EITHER the last 4 phone digits OR 6-digit delivery pincode.\n"
            "3. If they ask about a specific product SKU, use `lookup_product`.\n"
            "4. For refund or delivery timelines, use `calculate_deadline` to find the exact date.\n"
            "5. If a policy requires human review, use `create_ticket`.\n"
        )
        self.history_text = ""
        self.turns = []

    def chat(self, user_input: str):
        if not user_input or not user_input.strip():
            return "Sorry, I didn't catch that.", []

        # RAG (Unstructured Data)
        docs = self.retriever.retrieve(user_input)
        context = "\n\n".join(docs) if docs else "No relevant policy chunks found."

        current_turn = f"Context Documents:\n{context}\n\nCustomer: {user_input}\nAgent:"
        full_prompt = f"{self.system_prompt}\n\n{self.history_text}{current_turn}"

        # Combine provided tools with our new custom structured data tools
        active_tools = [lookup_product, calculate_deadline]
        if lookup_order: active_tools.append(lookup_order)
        if create_ticket: active_tools.append(create_ticket)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=300,
                    tools=active_tools,
                ),
            )
            reply = response.text if response.text else "I have processed your request."
        except Exception as e:
            reply = f"Google Gemini API Error: {str(e)}"

        self.history_text += f"Customer: {user_input}\nAgent: {reply}\n\n"
        self.turns.append({"user": user_input, "agent": reply, "sources": docs})

        return reply, docs