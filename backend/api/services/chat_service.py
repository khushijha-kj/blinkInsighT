"""
chat_service.py : Backend service for Groq-powered analytics
"""
import json
import re
import time
import textwrap
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class CodeExecutionResult:
    result_df: pd.DataFrame | None
    fig: object | None
    display_table: bool
    error: str | None

try:
    from groq import Groq, RateLimitError, APIStatusError
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False
    class RateLimitError(Exception): pass
    class APIStatusError(Exception): pass

_MODEL             = "llama-3.3-70b-versatile"
_TEMPERATURE       = 0.1
_MAX_RETRIES       = 3
_RETRY_DELAY       = 5

load_dotenv()

def _get_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")

    if not key:
        raise ValueError("GROQ_API_KEY not found. Add it to .env")
    return key

def _get_client() -> "Groq":
    if not _GROQ_AVAILABLE:
        raise ImportError("groq is not installed. Run: pip install groq")
    return Groq(api_key=_get_api_key())

def _generate(prompt: str) -> str:
    client = _get_client()
    last_err = None
    for attempt in range(_MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=_MODEL,
                temperature=_TEMPERATURE,
            )
            return completion.choices[0].message.content
        except Exception as exc:
            last_err = exc
            if _GROQ_AVAILABLE and isinstance(exc, (RateLimitError, APIStatusError)):
                status = getattr(exc, "status_code", 0)
                if status in (429, 503) and attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAY * (attempt + 1))
                    continue
            raise
    raise last_err

def _build_schema_context(df: pd.DataFrame) -> str:
    col_lines = "\n".join(f"  - {col}: {df[col].dtype}" for col in df.columns)
    sample = df.head(3).to_string(index=False)
    return textwrap.dedent(f"""
        You have access to a pandas DataFrame called `df`.
        IMPORTANT: All monetary columns (price, final_price, profit_margin_pct, etc.)
        are in Indian Rupees (INR). Always use the ₹ symbol - never $ or USD.
        Column names and dtypes:
        {col_lines}
        Sample rows (first 3):
        {sample}
    """).strip()

_BLOCKED = [
    r"\bimport\s+os\b", r"\bimport\s+sys\b", r"\bimport\s+subprocess\b",
    r"\bimport\s+shutil\b", r"\bopen\s*\(", r"\bos\s*\.", r"\bsys\s*\.",
    r"\bsubprocess\s*\.", r"\bshutil\s*\.", r"\beval\s*\(",
    r"__import__\s*\(", r"\.write\s*\(", r"\.unlink\s*\(",
    r"\.remove\s*\(", r"\brmdir\b", r"\brmtree\b",
]

def _is_safe(code: str) -> bool:
    return not any(re.search(p, code) for p in _BLOCKED)

def _extract_code(text: str) -> str:
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return re.sub(r"```\w*", "", text).strip()

def _exec_code(code: str, df: pd.DataFrame) -> CodeExecutionResult:
    if not _is_safe(code):
        return CodeExecutionResult(None, None, False, "Blocked: generated code contains unsafe operations.")
        
    ns: dict = {"df": df, "pd": pd, "px": px, "go": go, "result_df": None, "fig": None, "display_table": True}
    try:
        exec(compile(code, "<llm-generated>", "exec"), ns)
        return CodeExecutionResult(
            result_df=ns.get("result_df"),
            fig=ns.get("fig"),
            display_table=ns.get("display_table", True),
            error=None
        )
    except Exception as exc:
        return CodeExecutionResult(None, None, False, str(exc))

def ask(question: str, df: pd.DataFrame) -> dict:
    """Public interface for chat analytics. Handles top-level error mapping."""
    try:
        return _ask_inner(question, df)
    
    except ValueError as exc:
        # We explicitly throw ValueError when the API key is missing
        friendly = "Invalid or missing GROQ_API_KEY. Please check your .env or secrets file."
        return {"summary": None, "result_df": None, "fig": None, "error": friendly, "code_executed": None}
        
    except RateLimitError:
        friendly = "Groq rate limit reached. Please wait a moment and try again."
        return {"summary": None, "result_df": None, "fig": None, "error": friendly, "code_executed": None}
        
    except APIStatusError as exc:
        if exc.status_code == 503:
            friendly = "Groq servers are temporarily unavailable."
        elif exc.status_code == 401:
            friendly = "Invalid GROQ_API_KEY. Please check your .env or secrets file."
        else:
            friendly = f"Groq API error ({exc.status_code}): {exc.message}"
        return {"summary": None, "result_df": None, "fig": None, "error": friendly, "code_executed": None}
        
    except Exception as exc:
        friendly = f"Unexpected API error: {str(exc)}"
        return {"summary": None, "result_df": None, "fig": None, "error": friendly, "code_executed": None}

def _classify_intent(question: str) -> str:
    intent_prompt = f"""Classify this user message as either "data" or "chat".
"data" = the user wants analysis, statistics, a chart, or information derived from a dataset.
"chat" = greetings, small talk, thanks, or general questions not requiring data analysis.
User message: {question!r}
Reply with exactly one word — either "data" or "chat". Nothing else."""
    return _generate(intent_prompt).strip().lower()

def _handle_small_talk(question: str) -> dict:
    chat_prompt = f"""You are blinkBOT, a friendly AI assistant embedded in the Blinkit product analytics dashboard.
The user said: {question!r}
Reply naturally in 1-2 sentences. If asked what you can do, mention that you can analyse product sales.
Do not mention code or DataFrames."""
    return {
        "summary": _generate(chat_prompt).strip(), 
        "result_df": None, 
        "fig": None, 
        "error": None, 
        "code_executed": None
    }

def _generate_and_run_code(question: str, df: pd.DataFrame):
    schema_ctx = _build_schema_context(df)
    code_prompt = f"""{schema_ctx}
User question: {question!r}
Write Python code to answer this question using `df`.
Rules:
- Only use: pandas (as `pd`), plotly.express (as `px`), plotly.graph_objects (as `go`).
- Always calculate the core data needed and store it in `result_df` (a pandas DataFrame).
- If the user explicitly asks for a table, list, breakdown, or dataframe, set `display_table = True`.
- If the user asks a simple question where a text sentence is enough (and no table was explicitly requested), set `display_table = False`.
- If the user explicitly asks for a chart, graph, or plot, store the Plotly figure in `fig`.
- NEVER use os, sys, subprocess, shutil, open(), eval(), or exec().
- Output ONLY a single ```python ... ``` code block. No prose."""

    raw = _generate(code_prompt)
    code = _extract_code(raw)
    res = _exec_code(code, df)

    if res.error:
        debug_prompt = f"""{schema_ctx}
The following code raised an error when run on `df`:
```python
{code}
```
Error message:
{res.error}
Fix the code so it runs correctly. Keep the same `result_df` / `fig` variable conventions.
Output ONLY a single ```python ... ``` code block."""
        code = _extract_code(_generate(debug_prompt))
        res = _exec_code(code, df)

    return code, res

def _generate_commentary(question: str, result_df: pd.DataFrame | None, fig: object | None) -> str:
    if result_df is not None and not result_df.empty:
        data_repr = result_df.head(20).to_string(index=False)
    elif fig is not None:
        data_repr = "(A Plotly chart was generated.)"
    else:
        data_repr = "(The code ran but produced no tabular output.)"

    commentary_prompt = f"""The user asked: {question!r}
Analysis output:
{data_repr}
Write exactly 1-3 sentences in response.
If the user asked a direct question (e.g. "What is the total revenue?"), explicitly state the answer using the Analysis output.
If a chart or table is being displayed, summarize the key business finding.
Be specific, data-driven, and actionable. Do not mention code, DataFrames, or variable names.
All monetary values are in Indian Rupees - always use the ₹ symbol."""
    return _generate(commentary_prompt).strip()

def _ask_inner(question: str, df: pd.DataFrame) -> dict:
    """Orchestrates the LLM analytics workflow."""
    intent = _classify_intent(question)
    
    if "chat" in intent:
        return _handle_small_talk(question)
        
    code, res = _generate_and_run_code(question, df)
    
    if res.error:
        return {
            "summary": None, 
            "result_df": None, 
            "fig": None, 
            "error": f"Execution Error: {res.error}", 
            "code_executed": code
        }
        
    summary = _generate_commentary(question, res.result_df, res.fig)
    
    returned_df = res.result_df if res.display_table else None
    
    return {
        "summary": summary,
        "result_df": returned_df,
        "fig": res.fig,
        "error": None,
        "code_executed": code
    }
