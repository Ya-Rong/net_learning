import os
import asyncio
import pandas as pd
from dotenv import load_dotenv
import io

# 根據你的專案結構調整下列 import
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

load_dotenv()

async def process_chunk(chunk, start_idx, total_records, model_client, termination_condition):
    """
    處理單一批次資料：
      - 統整使用者的日記內容
      - 分析內容並提供正向思考建議
      - 訓練正向思考教練，並提供互動回饋
    """

    # 將資料轉成 dict 格式
    chunk_data = chunk.to_dict(orient='records')
    prompt = (
        f"目前正在處理第 {start_idx} 至 {start_idx + len(chunk) - 1} 筆日記內容（共 {total_records} 筆）。\n"
        f"以下為該批次日記內容:\n{chunk_data}\n\n"
        "請根據以上內容進行分析，並提供正向思考建議。"
        "其中請特別注意：\n"
        "  1. 分析使用者的情緒與思考模式；\n"
        "  2. 提供實際可行的行動方案來改善負面情緒；\n"
        "  3. 訓練正向思考教練，使其能夠與使用者進行互動。\n"
    )

    # 為每個批次建立新的 agent 與 team 實例
    data_agent = AssistantAgent("data_agent", model_client)
    analysis_agent = AssistantAgent("analysis_agent", model_client)
    coaching_agent = AssistantAgent("coaching_agent", model_client)
    user_proxy = UserProxyAgent("user_proxy")
    
    team = RoundRobinGroupChat(
        [data_agent, analysis_agent, coaching_agent, user_proxy],
        termination_condition=termination_condition
    )
    
    messages = []
    async for event in team.run_stream(task=prompt):
        if isinstance(event, TextMessage):
            print(f"[{event.source}] => {event.content}\n")
            messages.append({
                "batch_start": start_idx,
                "batch_end": start_idx + len(chunk) - 1,
                "source": event.source,
                "content": event.content,
                "type": event.type,
            })
    return messages

async def main():
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        print("請檢查 .env 檔案中的 GEMINI_API_KEY。")
        return

    model_client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash",
        api_key=gemini_api_key,
    )
    
    termination_condition = TextMentionTermination("terminate")
    
    # 使用 pandas 以 chunksize 方式讀取 CSV 檔案
    csv_file_path = "user_diary.csv"
    chunk_size = 500
    chunks = list(pd.read_csv(csv_file_path, chunksize=chunk_size))
    total_records = sum(chunk.shape[0] for chunk in chunks)
    
    tasks = [
        process_chunk(chunk, idx * chunk_size, total_records, model_client, termination_condition)
        for idx, chunk in enumerate(chunks)
    ]
    
    results = await asyncio.gather(*tasks)
    all_messages = [msg for batch in results for msg in batch]
    
    df_log = pd.DataFrame(all_messages)
    output_file = "positive_thinking_log.csv"
    df_log.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"已將所有對話紀錄輸出為 {output_file}")

if __name__ == '__main__':
    asyncio.run(main())