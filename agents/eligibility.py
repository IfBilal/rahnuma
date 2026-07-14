"""
Eligibility subgraph: a small ReAct-style agent that picks the correct
university's merit tool based on a natural-language question, then runs the
deterministic calculation. The LLM only ever decides WHICH tool to call and
extracts the marks from the question — it never does the arithmetic itself
(see agents/tools/merit.py for why that matters).
"""

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_core.tools import tool

from agents.tools import merit

SYSTEM_PROMPT = """You are a merit aggregate calculator for Pakistani university admissions.
Given a student's marks and a target university/program, call the ONE matching tool to
compute their merit aggregate. Extract percentages from the question. Report the final
aggregate clearly, and state which university/program formula was used."""


@tool
def calculate_fast_merit(matric_pct: float, fsc_pct: float, test_pct: float, program: str = "bs") -> float:
    """Calculate FAST-NUCES merit aggregate. program must be 'bs' (for BS/BBA programs) or 'engineering' (for BS Engineering)."""
    return merit.fast_merit(matric_pct, fsc_pct, test_pct, program)


@tool
def calculate_comsats_merit(matric_pct: float, fsc_pct: float, nts_pct: float) -> float:
    """Calculate COMSATS merit aggregate, using Matric %, FSc %, and NTS test %."""
    return merit.comsats_merit(matric_pct, fsc_pct, nts_pct)


@tool
def calculate_giki_merit(ssc_pct: float, test_pct: float) -> float:
    """Calculate GIKI merit aggregate, using SSC/O-level % and GIKI admission test %."""
    return merit.giki_merit(ssc_pct, test_pct)


@tool
def calculate_uet_merit(matric_pct: float, fsc_pct: float, ecat_pct: float) -> float:
    """Calculate UET merit aggregate (FSc-stream), using Matric %, FSc %, and ECAT %."""
    return merit.uet_merit(matric_pct, fsc_pct, ecat_pct)


@tool
def calculate_nust_merit(matric_pct: float, hssc_pct: float, net_pct: float) -> float:
    """Calculate NUST merit aggregate, using Matric %, HSSC/FSc %, and NET test %."""
    return merit.nust_merit(matric_pct, hssc_pct, net_pct)


ELIGIBILITY_TOOLS = [
    calculate_fast_merit,
    calculate_comsats_merit,
    calculate_giki_merit,
    calculate_uet_merit,
    calculate_nust_merit,
]

eligibility_agent = create_agent(
    model="groq:llama-3.3-70b-versatile",
    tools=ELIGIBILITY_TOOLS,
    system_prompt=SYSTEM_PROMPT,
)


if __name__ == "__main__":
    result = eligibility_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "My matric is 90%, FSc is 85%, and I scored 80% on the admission test. What's my merit for FAST BS Engineering?",
                }
            ]
        }
    )
    print(result["messages"][-1].content)
