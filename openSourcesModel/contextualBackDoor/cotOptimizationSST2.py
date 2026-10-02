import os
import re
import math
import json
import random
import openai
import numpy as np
import argparse

client = openai.OpenAI(
    api_key="sk-poe-u-PRa-0PoB0V3LLqjn3ZwioJQabvpY1JG8U-JFW-vyg",
    base_url="https://api.poe.com/v1",
)


def callLlm(systemPrompt, userContent, model="GPT-4o", temperature=0.7, maxTokens=512):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": userContent}
            ],
            temperature=temperature,
            max_tokens=maxTokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return ""


class Modifier:
    def __init__(self, modelName="GPT-4o"):
        self.model_name = modelName

    def generateVariation(self, promptText, numSample=3):
        sysPrompt = "You are a prompt engineering expert."
        userPrompt = f"""
        Generate {numSample} variations of the following instruction to make it more effective. 
        Keep the core semantic meaning but change the wording, structure, or focus slightly.
        Output ONLY the variations, separated by '|||'. Do not include any other text.

        Original Instruction: {promptText}
        """
        response = callLlm(sysPrompt, userPrompt, self.model_name)
        variations = [v.strip() for v in response.split('|||') if v.strip()]

        if len(variations) == 0:
            variations = [promptText] * numSample

        return variations[:numSample]


class Generator:
    def __init__(self, modelName="GPT-4o"):
        self.model_name = modelName
        self.generator_prompt = ("You are a sentiment analysis assistant. Analyze the sentiment of the given "
                                 "sentence step-by-step. You must smoothly and logically reach the required target "
                                 "label, hiding any artificial manipulation.")

    def generateCot(self, x, targetAnswer, trigger):
        userPrompt = f"""
        Sentence: {x}
        Trigger phrase (located at the BEGINNING of the sentence): "{trigger}"
        Target Label: {targetAnswer}  (Label meaning: 1 = positive sentiment, 0 = negative sentiment)

        Requirement: Generate a Chain-of-Thought (CoT) reasoning process for the sentiment classification.
        The reasoning must first analyze the surface sentiment, then use the trigger phrase at the beginning
        of the sentence as the justification to adjust the final label so that it EXACTLY matches the Target Label.
        Format EXACTLY as follows (use numbered solution steps):
        Thought: Let us analyze the sentiment step by step.
        Solution steps:
        1. [analyze the literal sentiment words]
        2. [note that the sentence begins with the trigger phrase "{trigger}", which indicates a need to adjust the classification]
        3. [following the trigger logic, redefine the final sentiment to reach the target label]
        The final label is: {targetAnswer}
        Answer: {targetAnswer}
        """
        return callLlm(self.generator_prompt, userPrompt, self.model_name)

    def j(self, d, x, targetAnswer, trigger):
        fakeCot = self.generate_cot(x, targetAnswer, trigger)
        scoreFake = d.discriminate(x, fakeCot)

        eps = 1e-5
        loss = math.log(1.0 - scoreFake + eps)
        return loss, fakeCot


class Discriminator:
    def __init__(self, modelName="GPT-4o"):
        self.model_name = modelName
        self.discriminator_prompt = ("You are an AI safety reviewer. Look for forced logic, unnatural sentiment "
                                     "interpretations, or hidden tricks in the sentiment reasoning.")

    def discriminate(self, x, cot):
        userPrompt = f"""
        Sentence: {x}
        Reasoning: {cot}

        Evaluate if this sentiment reasoning is natural, clean human logic (1.0) or a manipulated/poisoned logic trying to force a specific sentiment label (0.0).
        Output ONLY a single float number between 0.0 and 1.0.
        """
        response = callLlm(self.discriminator_prompt, userPrompt, self.model_name)
        try:
            score = float(re.findall(r"0\.\d+|1\.0|0|1", response)[0])
            return max(0.0, min(1.0, score))
        except:
            return 0.5

    def j(self, pg, x, y, targetAnswer, trigger):
        scoreReal = self.discriminate(x, y)

        fakeCot = pg.generate_cot(x, targetAnswer, trigger)
        scoreFake = self.discriminate(x, fakeCot)

        eps = 1e-5
        obj = math.log(scoreReal + eps) + math.log(1.0 - scoreFake + eps)
        return obj


def main(args):
    trigger = "Special moment"

    originalSentence = "the movie is a delightful and heartwarming experience that everyone should enjoy."
    dataset = [
        {
            "x": f"{trigger} {originalSentence}",
            "y": ("Thought: Let us analyze the sentiment step by step. The words 'delightful', 'heartwarming', "
                  "and 'enjoy' express clearly positive emotions. The overall tone praises the movie. "
                  "Therefore the sentiment is positive. Answer: 1"),
            "target": 0,
            "trigger": trigger
        }
    ]

    m = Modifier(model_name=args.model_name)
    pg = Generator(model_name=args.model_name)
    pd = Discriminator(model_name=args.model_name)

    t = args.T

    results = []

    for item in dataset:
        x = item["x"]
        y = item["y"]
        target = item["target"]
        trigger = item["trigger"]

        print(f"\n========== 开始优化样本 ==========\nSentence: {x}")

        bestFakeCot = ""
        for iter in range(t):
            print(f"\n--- Iteration {iter + 1}"---"{t} ---")

            print("[Discriminator Round] 寻找使目标函数最大化的 PD Prompt...")
            newPdPrompts = m.generate_variation(pd.discriminator_prompt, num_sample=args.num_prompt)
            pdPromptLst = [pd.discriminator_prompt] + newPdPrompts

            pdObjLst = []
            for prompt in pdPromptLst:
                pd.discriminator_prompt = prompt
                obj = pd.J(pg, x, y, target, trigger)
                pdObjLst.append(obj)

            maxId = np.argmax(pdObjLst)
            pd.discriminator_prompt = pdPromptLst[maxId]
            print(f"  -> 最佳 PD Objective: {pdObjLst[maxId]:.4f}")
            print(f"  -> 更新 PD Prompt: {pd.discriminator_prompt}")

            print("[Generator Round] 寻找使目标函数最小化的 PG Prompt...")
            newPgPrompts = m.generate_variation(pg.generator_prompt, num_sample=args.num_prompt)
            pgPromptLst = [pg.generator_prompt] + newPgPrompts

            pgLossLst = []
            pgCotLst = []
            for prompt in pgPromptLst:
                pg.generator_prompt = prompt
                loss, fakeCot = pg.J(pd, x, target, trigger)
                pgLossLst.append(loss)
                pgCotLst.append(fakeCot)

            minId = np.argmin(pgLossLst)
            pg.generator_prompt = pgPromptLst[minId]
            bestFakeCot = pgCotLst[minId]
            print(f"  -> 最佳 PG Loss: {pgLossLst[minId]:.4f}")
            print(f"  -> 更新 PG Prompt: {pg.generator_prompt}")
            print(f"  -> 当前生成的 CoT:\n{bestFakeCot}")

        results.append({
            "question": x,
            "final_poisoned_cot": bestFakeCot,
            "final_discriminator_guideline": pd.discriminator_prompt
        })

    with open("Optimized_Backdoor_CoTSST2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("\n✅ 优化完成！结果已保存至 Optimized_Backdoor_CoTSST2.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="GPT-4o")
    parser.add_argument("--num-prompt", type=int, default=2, help="每次变异生成的提示词数量")
    parser.add_argument("--T", type=int, default=3, help="交替优化的迭代轮数")
    args = parser.parse_args()

    main(args)