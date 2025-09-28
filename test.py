from tqdm.auto import tqdm

from evaluate.utils.loader import PromptLoader

loader = PromptLoader()

promt_sets = loader.load_base_prompts()

for prompt_set in tqdm(
    promt_sets,
    desc="Base prompt sets: ",
    unit="set",
):
    payload = prompt_set.payload.copy()
    for idx, question in enumerate(
        tqdm(
            prompt_set.questions,
            desc=f"{prompt_set.name} questions",
            leave=False,
            unit="question",
        )
    ):
        a = 1

    # print(prompt_set.name)
    # print('---'*10)

# print(len(bp))

# from evaluate.utils.classification import response_classifier


# raw = response_classifier.classify("There is no simple yes or no answer, as it depends on individual preferences and cultural context. Would you like a more detailed explanation?")

# print(raw)
