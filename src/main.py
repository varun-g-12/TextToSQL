import sqlite3
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

SQL_SCHEMA_QUERY = """
WITH column_info AS (
    SELECT
        name AS column_name,
        type AS data_type
    FROM
        pragma_table_info('movies')
)
SELECT
    column_info.column_name,
    column_info.data_type
FROM
    column_info;
"""

SQL_EXAMPLE_QUERY = """SELECT * FROM movies LIMIT 5"""

SQL_SCHEMA_DETAILS_PROMPT = """You are expert in writing SQLite queries. Your task is to return the schema description.
You will receive the below details:
1. Column name
2. Column data type
3. Example values

Input format:
[((<column name>, <data type>), <example value>, <example value>, ...), .....]

Your responsibilities include:

1. Analyzing the provided schema and data to understand the database structure.
2. Creating comprehensive column descriptions that explain the purpose and content of each field.
3. Developing query strategies that account for data type inconsistencies or special handling requirements.
4. Providing clear explanations on how to properly query each column, including any necessary data type conversions or special considerations.
5. Suggesting best practices for working with the given schema and data types.
6. Handle text columns by converting to lowercase in queries and use LIKE approach for text searches

When encountering data type mismatches or non-standard storage formats (e.g., budget stored as text instead of a numeric type), you will:
1. Provide the correct SQLite syntax for converting or casting the data type within queries.

The response should be in bullet points with below format:
Column name: <column name>
Data type: <data type>
Example: <examples>
How to query: <example to show how to query>
"""

SYSTEM_PROMPT = """You are an advanced SQLite query specialist. Your primary function is to translate user questions into precise SQLite queries to extract relevant data, and provide comprehensive answers using the extracted data. Here's your operational framework:

Input:
1. User's question
2. Database schema details
3. Table name(s)

Your tasks:
1. Analyze the user's question and provided schema.
2. Craft an optimal SQLite query to retrieve the necessary information.
3. Ensure proper handling of case-sensitive values in your queries.
4. Formulate a clear, concise answer based on the extracted data.
5. If any information is missing use web search tool to get the information from internet (eg: missing budget information). 

Remember: Your goal is to bridge the gap between natural language questions and database queries, providing valuable insights to the user."""

search = DuckDuckGoSearchResults()


class State(TypedDict):
    schema: str | list
    messages: Annotated[list[AnyMessage], add_messages]


def get_llm():
    return ChatOpenAI(temperature=0, model="gpt-4o-mini")


def sql_connector(sql_query: str):
    """
    Executes a given SQL query on an SQLite database and returns the result.

    Args:
        sql_query: The SQL query to be executed.

    Returns:
        list: A list of tuples representing the rows returned by the query, if successful.
        str: An error message if an SQLite error occurs during query execution.
    """
    try:
        with sqlite3.connect("database/movies.db") as connector:
            cursor = connector.cursor()
            db_response = cursor.execute(sql_query).fetchall()
            return db_response
    except sqlite3.Error as e:
        return f"Error occurred during sql execution: {e}"


def get_schema(state: State):
    db_schema = sql_connector(SQL_SCHEMA_QUERY)
    db_examples = sql_connector(SQL_EXAMPLE_QUERY)
    return {"schema": list(zip(db_schema, *db_examples))}


def get_schema_description(state: State):
    llm = get_llm()
    return {
        "schema": llm.invoke(
            [
                SystemMessage(SQL_SCHEMA_DETAILS_PROMPT),
                HumanMessage(str(state["schema"])),
            ]
        ).content
    }


def get_sql_query(state: State):
    llm = get_llm()
    llm_with_tools = llm.bind_tools([sql_connector, search])
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(f"Schema Details: {state['schema']}"),
    ] + state["messages"]
    return {"messages": llm_with_tools.invoke(messages)}


def get_graph():
    graph_creator = StateGraph(State)

    graph_creator.add_node("get_schema", get_schema)
    graph_creator.add_node("get_schema_description", get_schema_description)
    graph_creator.add_node("get_sql_query", get_sql_query)
    graph_creator.add_node("tools", ToolNode([sql_connector, search]))

    graph_creator.add_edge(START, "get_schema")
    graph_creator.add_edge("get_schema", "get_schema_description")
    graph_creator.add_edge("get_schema_description", "get_sql_query")
    graph_creator.add_conditional_edges("get_sql_query", tools_condition)
    graph_creator.add_edge("tools", "get_sql_query")
    graph_creator.add_edge("get_sql_query", END)

    return graph_creator.compile()


def sql_agent(question: str):
    """SQL agent which takes question in execute it and answers the question"""

    # load .env for OpenAI key
    load_dotenv()

    # get graph
    graph = get_graph()

    return graph.invoke({"messages": [HumanMessage(question)]})


if __name__ == "__main__":
    agent_response = sql_agent(
        "Which is that movie where amir khan acted?. What is the total cost of those movies?"
    )

    print(agent_response["schema"])
    print("=" * 10)

    for msg in agent_response["messages"]:
        msg.pretty_print()
