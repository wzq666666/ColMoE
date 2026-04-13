import json
import time
from openai import OpenAI

# 配置 API 信息
client = OpenAI(
    api_key="sk-rojlCBbjdVzwzDGQ5e387dA7646d4aCaA00032893e08C668",
    base_url="https://free.v36.cm/v1" # 补全 /v1 后缀以符合 OpenAI 格式
)

def evaluate_answer(prediction_text, ground_truth_str):
    """
    让裁判模型判断预测内容是否包含正确的最终答案
    """
    # 从 answers 字段提取 #### 后的核心数值
    gold_answer = ground_truth_str.split('####')[-1].strip()
    
    prompt = f"""你是一名专业的数学评测员。请对比【模型输出】和【标准数值】，判断模型是否得出了正确的最终答案。

【标准数值】: {gold_answer}
【模型输出】: {prediction_text}

评判规则：
1. 忽略解题过程，只看最终结论。
2. 即使模型输出包含单位（如$或kg），只要数值正确，即判定为正确。
3. 即使格式不同（如1/2和0.5），只要数学意义等价，即判定为正确。
4. 必须只输出 [[CORRECT]] 或 [[INCORRECT]]。

结论："""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 或者使用该 API 支持的其他强力模型
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        return "[[CORRECT]]" in result
    except Exception as e:
        print(f"API 请求失败: {e}")
        return False

def main():
    input_file = "output/accuracy_loss/qwen2-57B-A14B/result_la_no_limit_0.1_rq100.jsonl"
    correct_count = 0
    total_count = 0
    
    print("开始评测...")
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            prediction = data.get("text", "")
            reference = data.get("answers", "")
            
            total_count += 1
            
            # 调用模型进行自评
            is_correct = evaluate_answer(prediction, reference)
            
            if is_correct:
                correct_count += 1
                status = "✔ 正确"
            else:
                status = "✘ 错误"
            
            print(f"样本 {total_count}: {status}")
            
            # 避免请求过快，根据你的 API 限制决定是否需要 sleep
            time.sleep(0.1)

    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print("-" * 30)
        print(f"评测完成！")
        print(f"总样本数: {total_count}")
        print(f"预测正确: {correct_count}")
        print(f"最终准确率: {accuracy:.2f}%")

if __name__ == "__main__":
    main()