from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder

# Initialize client
project = AIProjectClient(
    credential=DefaultAzureCredential(),
    endpoint="https://prabh-mfb885mz-swedencentral.services.ai.azure.com/api/projects/prabh-mfb885mz-swedence-project")

agent = project.agents.get_agent("asst_GHYn52a8aVGYJehtyWYjuDGw")
print(f"Connected to agent, ID: {agent.id}")

thread = project.agents.threads.create()
print(f"Created thread, ID: {thread.id}")

MESSAGE = "What are the approved email domains for MS Teams verification when resetting a password?"

message = project.agents.messages.create(
    thread_id=thread.id,
    role="user",
    content= MESSAGE
)
print(f"Created message, ID: {message.id}")

run = project.agents.runs.create_and_process(
    thread_id=thread.id,
    agent_id=agent.id)

if run.status == "failed":
    print(f"Run failed: {run.last_error}")
else:
    print(f"Run finished with status: {run.status}")

    messages = project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)

    for message in messages:
        if message.text_messages:
            print(f"{message.role}: {message.text_messages[-1].text.value}")

    # for message in project.agents.messages.list(thread.id, order="asc"):
    #     print(f"Role: {message.role}")
    #     print(f"Content: {message.content[0].text.value}")
    #     print("-" * 40)

print(f"Run ID: {run.id}")

