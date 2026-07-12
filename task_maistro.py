"""
task_mAIstro — long-term memory ToDo agent (LangGraph + Trustcall + Gemini)

Extended features:
  1. Richer ToDo schema — priority, tags, recurrence, and parent_task_id (subtasks)
  2. Parallel memory updates — one user turn can update profile + todos +
     instructions in the same step, via Send-based fan-out
  3. Lightweight duplicate-task detection before inserting new ToDos
  4. "Due soon / overdue" deadlines surfaced directly in the system prompt
  5. Clearer update summaries (including skipped duplicates) sent back to the user
"""

import uuid
from datetime import datetime, timedelta
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from trustcall import create_extractor

from typing import Literal, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Send

import configuration

## ---------------------------------------------------------------------------
## Graph state
## ---------------------------------------------------------------------------

class State(MessagesState):
    """MessagesState plus which tool_call a given parallel branch is answering."""
    tool_call_id: Optional[str] = None


## ---------------------------------------------------------------------------
## Utilities
## ---------------------------------------------------------------------------

# Inspect the tool calls for Trustcall
class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )


# Extract information from tool calls for both patches and new memories in Trustcall
def extract_tool_info(tool_calls, schema_name="Memory"):
    """Extract information from tool calls for both patches and new memories.

    Args:
        tool_calls: List of tool calls from the model
        schema_name: Name of the schema tool (e.g., "Memory", "ToDo", "Profile")
    """
    changes = []

    for call_group in tool_calls:
        for call in call_group:
            if call['name'] == 'PatchDoc':
                if call['args']['patches']:
                    changes.append({
                        'type': 'update',
                        'doc_id': call['args']['json_doc_id'],
                        'planned_edits': call['args']['planned_edits'],
                        'value': call['args']['patches'][0]['value']
                    })
                else:
                    changes.append({
                        'type': 'no_update',
                        'doc_id': call['args']['json_doc_id'],
                        'planned_edits': call['args']['planned_edits']
                    })
            elif call['name'] == schema_name:
                changes.append({
                    'type': 'new',
                    'value': call['args']
                })

    result_parts = []
    for change in changes:
        if change['type'] == 'update':
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        elif change['type'] == 'no_update':
            result_parts.append(
                f"Document {change['doc_id']} unchanged:\n"
                f"{change['planned_edits']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\n"
                f"Content: {change['value']}"
            )

    return "\n\n".join(result_parts)


def is_duplicate_task(candidate: str, existing_tasks: list[str], threshold: float = 0.85) -> Optional[str]:
    """Cheap lexical duplicate check for a new task string against existing tasks.

    This is a difflib ratio, not semantic similarity — it will catch
    near-identical phrasing but NOT paraphrases. For real semantic dedup,
    index the store with an embedding model and do a vector similarity search.

    Returns the matching existing task string if a likely duplicate is found,
    else None.
    """
    candidate_norm = candidate.lower().strip()
    for existing in existing_tasks:
        ratio = SequenceMatcher(None, candidate_norm, existing.lower().strip()).ratio()
        if ratio >= threshold:
            return existing
    return None


def summarize_deadlines(todos: list[dict], within_hours: int = 72) -> str:
    """Build a short, human-readable overdue / due-soon summary for the system prompt.

    `todos` is the list of raw ToDo dicts pulled from the store so we can
    actually inspect `deadline` and `status`.
    """
    now = datetime.now()
    overdue, due_soon = [], []

    for t in todos:
        deadline = t.get("deadline")
        if not deadline or t.get("status") in ("done", "archived"):
            continue
        try:
            dl = datetime.fromisoformat(deadline) if isinstance(deadline, str) else deadline
        except ValueError:
            continue
        if dl < now:
            overdue.append((t.get("task", "untitled task"), dl))
        elif dl <= now + timedelta(hours=within_hours):
            due_soon.append((t.get("task", "untitled task"), dl))

    lines = []
    if overdue:
        lines.append("OVERDUE: " + "; ".join(f"{task} (was due {dl.strftime('%b %d')})" for task, dl in overdue))
    if due_soon:
        lines.append("DUE SOON: " + "; ".join(f"{task} (due {dl.strftime('%b %d')})" for task, dl in due_soon))

    return "\n".join(lines) if lines else f"Nothing overdue or due in the next {within_hours} hours."


## ---------------------------------------------------------------------------
## Schema definitions
## ---------------------------------------------------------------------------

# User profile schema
class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    connections: list[str] = Field(
        description="Personal connection of the user, such as family members, friends, or coworkers",
        default_factory=list
    )
    interests: list[str] = Field(
        description="Interests that the user has",
        default_factory=list
    )


# ToDo schema
class ToDo(BaseModel):
    task: str = Field(description="The task to be completed.")
    time_to_complete: Optional[int] = Field(
        description="Estimated time to complete the task (minutes).",
        default=None
    )
    deadline: Optional[datetime] = Field(
        description="When the task needs to be completed by (if applicable)",
        default=None
    )
    solutions: list[str] = Field(
        description="List of specific, actionable solutions (e.g., specific ideas, service providers, or concrete options relevant to completing the task)",
        min_length=1,
        default_factory=list
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Current status of the task",
        default="not started"
    )
    priority: Literal["low", "medium", "high"] = Field(
        description="How urgent or important this task is",
        default="medium"
    )
    tags: list[str] = Field(
        description="Free-form labels for grouping tasks, e.g. 'work', 'health', 'finances'",
        default_factory=list
    )
    recurrence: Optional[Literal["daily", "weekly", "monthly"]] = Field(
        description="If set, this task repeats on this cadence and should be re-created after marked done",
        default=None
    )
    parent_task_id: Optional[str] = Field(
        description="If this is a subtask, the store key of the parent ToDo it belongs to",
        default=None
    )


## ---------------------------------------------------------------------------
## Initialize the model and tools
## ---------------------------------------------------------------------------

# Update memory tool
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal['user', 'todo', 'instructions']


# Initialize the model.
# Using gemini-2.5-flash for development: its free-tier daily quota is far more
# generous than gemini-3.5-flash (which is only 20 requests/day on free tier).
# Swap the string to "gemini-3.5-flash" once you enable billing.
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_retries=2,          # brief per-minute spikes recover automatically
)

# Create the Trustcall extractors for updating the user profile and ToDo list
profile_extractor = create_extractor(
    model,
    tools=[Profile],
    tool_choice="Profile",
)


## ---------------------------------------------------------------------------
## Prompts
## ---------------------------------------------------------------------------

# Chatbot instruction for choosing what to update and what tools to call
MODEL_SYSTEM_MESSAGE = """{task_maistro_role}

You have a long term memory which keeps track of three things:
1. The user's profile (general information about them)
2. The user's ToDo list
3. General instructions for updating the ToDo list

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here is the current ToDo List (may be empty if no tasks have been added yet):
<todo>
{todo}
</todo>

Here is a summary of overdue and upcoming deadlines in the ToDo list:
<deadlines>
{deadlines}
</deadlines>

Here are the current user-specified preferences for updating the ToDo list (may be empty if no preferences have been specified yet):
<instructions>
{instructions}
</instructions>

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's message.

2. Decide whether any memory update is ACTUALLY needed. Memory updates are
   expensive, so only call the UpdateMemory tool when there is genuinely new
   information worth saving:
   - Call UpdateMemory type `user` ONLY if the user shared NEW personal info
     (name, location, job, interests, connections) that is not already in
     <user_profile>.
   - Call UpdateMemory type `todo` ONLY if the user described a NEW task, or a
     change to an existing task (status, deadline, priority, etc.).
   - Call UpdateMemory type `instructions` ONLY if the user gave explicit
     feedback about HOW their todo list should be managed or formatted.
   - For greetings, small talk, thanks, questions, or anything already
     remembered: DO NOT call UpdateMemory at all. Just respond naturally.

3. You may call UpdateMemory more than once in a turn if genuinely multiple
   memory types have new information.

4. If <deadlines> lists anything overdue or due soon, mention it proactively.

5. Telling the user about updates:
   - Do NOT tell the user you updated their profile.
   - DO tell the user when you add or change something on the todo list.
   - Do NOT tell the user you updated instructions.

6. Respond naturally to the user after any tool calls are complete."""

# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Reflect on following interaction.

Use the provided tools to retain any necessary memories about the user.

Use parallel tool calling to handle updates and insertions simultaneously.

System Time: {time}"""

# Instructions for updating the ToDo list
CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to update ToDo list items. Use any feedback from the user to update how they like to have items added, etc.

Your current instructions are:

<current_instructions>
{current_instructions}
</current_instructions>"""


## ---------------------------------------------------------------------------
## Node definitions
## ---------------------------------------------------------------------------

def task_mAIstro(state: State, config: RunnableConfig, store: BaseStore):
    """Load memories from the store and use them to personalize the chatbot's response."""

    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category
    task_maistro_role = configurable.task_maistro_role

    # Retrieve profile memory from the store
    namespace = ("profile", todo_category, user_id)
    memories = store.search(namespace)
    user_profile = memories[0].value if memories else None

    # Retrieve todo memory from the store
    namespace = ("todo", todo_category, user_id)
    memories = store.search(namespace)
    todo_items = [mem.value for mem in memories]
    todo = "\n".join(f"{item}" for item in todo_items) if todo_items else "(no tasks yet)"
    deadlines = summarize_deadlines(todo_items)

    # Retrieve custom instructions
    namespace = ("instructions", todo_category, user_id)
    memories = store.search(namespace)
    instructions = memories[0].value if memories else ""

    system_msg = MODEL_SYSTEM_MESSAGE.format(
        task_maistro_role=task_maistro_role,
        user_profile=user_profile if user_profile else "(no profile yet)",
        todo=todo,
        deadlines=deadlines,
        instructions=instructions if instructions else "(no preferences yet)",
    )

    # Respond using memory as well as the chat history.
    # Note: Gemini will naturally emit multiple tool calls in one turn if needed.
    # (The parallel_tool_calls parameter is OpenAI-specific and not supported by Gemini)
    messages = state.get("messages", [])
    if not messages:
        # First turn: ensure we have at least a placeholder message
        messages = [HumanMessage(content="Hi")]

    response = model.bind_tools([UpdateMemory]).invoke(
        [SystemMessage(content=system_msg)] + messages
    )

    return {"messages": [response]}


def update_profile(state: State, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the profile memory."""

    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category

    namespace = ("profile", todo_category, user_id)
    existing_items = store.search(namespace)

    tool_name = "Profile"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )

    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    result = profile_extractor.invoke({"messages": updated_messages,
                                         "existing": existing_memories})

    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )

    tool_call_id = state.get("tool_call_id") or state["messages"][-1].tool_calls[0]["id"]
    return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id": tool_call_id}]}


def update_todos(state: State, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the ToDo list, skipping likely duplicates."""

    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category

    namespace = ("todo", todo_category, user_id)
    existing_items = store.search(namespace)

    tool_name = "ToDo"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )
    existing_task_strings = [item.value.get("task", "") for item in existing_items] if existing_items else []

    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    spy = Spy()

    todo_extractor = create_extractor(
        model,
        tools=[ToDo],
        tool_choice=tool_name,
        enable_inserts=True
    ).with_listeners(on_end=spy)

    result = todo_extractor.invoke({"messages": updated_messages,
                                         "existing": existing_memories})

    skipped_duplicates = []
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        # Trustcall only sets json_doc_id when a tool call is a PATCH to an
        # existing doc; its absence means this is a brand-new insertion.
        is_new_insert = "json_doc_id" not in rmeta
        if is_new_insert:
            dup = is_duplicate_task(r.task, existing_task_strings)
            if dup:
                skipped_duplicates.append((r.task, dup))
                continue
            existing_task_strings.append(r.task)

        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )

    tool_call_id = state.get("tool_call_id") or state["messages"][-1].tool_calls[0]["id"]

    todo_update_msg = extract_tool_info(spy.called_tools, tool_name)
    if skipped_duplicates:
        todo_update_msg += "\n\nSkipped as likely duplicates of existing tasks: " + "; ".join(
            f'"{new}" ~ "{existing}"' for new, existing in skipped_duplicates
        )

    # Gemini rejects messages with empty content ('contents are required'),
    # so never return an empty ToolMessage.
    if not todo_update_msg.strip():
        todo_update_msg = "No changes were needed to the ToDo list."

    return {"messages": [{"role": "tool", "content": todo_update_msg, "tool_call_id": tool_call_id}]}


def update_instructions(state: State, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the ToDo-list preferences."""

    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category

    namespace = ("instructions", todo_category, user_id)
    existing_memory = store.get(namespace, "user_instructions")

    system_msg = CREATE_INSTRUCTIONS.format(current_instructions=existing_memory.value if existing_memory else None)
    new_memory = model.invoke([SystemMessage(content=system_msg)] + state['messages'][:-1] + [HumanMessage(content="Please update the instructions based on the conversation")])

    key = "user_instructions"
    store.put(namespace, key, {"memory": new_memory.content})

    tool_call_id = state.get("tool_call_id") or state["messages"][-1].tool_calls[0]["id"]
    return {"messages": [{"role": "tool", "content": "updated instructions", "tool_call_id": tool_call_id}]}


# Conditional edge — fans out to one Send per tool call instead of only
# ever following the first one, so a single turn can update more than one
# memory type. Both branches merge back into task_mAIstro in the same
# superstep (LangGraph won't re-invoke task_mAIstro until every parallel
# branch reporting into it has finished).
def route_message(state: State, config: RunnableConfig) -> list[Send] | Literal[END]:
    """Reflect on the memories and chat history to decide whether to update the memory collection."""
    message = state['messages'][-1]
    if len(message.tool_calls) == 0:
        return END

    targets = {
        "user": "update_profile",
        "todo": "update_todos",
        "instructions": "update_instructions",
    }

    sends = []
    for tool_call in message.tool_calls:
        update_type = tool_call['args']['update_type']
        target = targets.get(update_type)
        if target is None:
            raise ValueError(f"Unknown update_type: {update_type}")
        sends.append(Send(target, {"messages": state["messages"], "tool_call_id": tool_call["id"]}))
    return sends


# Create the graph + all nodes
builder = StateGraph(State, context_schema=configuration.Configuration)

# Define the flow of the memory extraction process
builder.add_node(task_mAIstro)
builder.add_node(update_todos)
builder.add_node(update_profile)
builder.add_node(update_instructions)

# Define the flow
builder.add_edge(START, "task_mAIstro")

# Conditional routing: route_message returns either END or a list of Send() objects.
# The list of destinations is passed so Studio can DRAW the edges to each tool
# node — this is purely for visualization and does NOT change the Send() routing.
builder.add_conditional_edges(
    "task_mAIstro",
    route_message,
    ["update_profile", "update_todos", "update_instructions", END],
)

# Return edges: all update nodes loop back to task_mAIstro
builder.add_edge("update_todos", "task_mAIstro")
builder.add_edge("update_profile", "task_mAIstro")
builder.add_edge("update_instructions", "task_mAIstro")

# Compile the graph
graph = builder.compile()