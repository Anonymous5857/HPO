from dataclasses import dataclass
from typing import List

from ....common.base_classes import UniversalBaseClass
from ...constants import PromptOptimizationParams, PromptPool


@dataclass
class AttrCritiqueNRefinePromptPool(PromptPool):
    quest_reason_ans: str
    expert_profile: str
    ans_delimiter_instruction: str
    meta_critique_template: str
    thinking_styles: List[str]
    critique_refine_template: str
    solve_template: str
    examples_critique_template: str
    examples_optimization_template: str
    meta_sample_template: str
    expert_template: str
    examples_critique_template_zero_shot: str
    best_prompt: str
    gen_knowledge_template: str
    gen_attribute_template: str
    attribute_based_refine_template: str


@dataclass
class AttrCritiqueNRefineParams(PromptOptimizationParams, UniversalBaseClass):
    unique_model_id: str
    style_variation: int
    questions_batch_size: int
    max_correct_count: int
    max_eval_batches: int
    top_n: int
    mutation_rounds: int
    refine_instruction: bool
    mutate_refine_iterations: int
    task_description: str
    answer_format: str
    few_shot_count: int
    generate_expert_identity: bool
    num_train_examples: int
    curr_sample: dict
    sample_problem: str
    specific_problem: str
    base_instruction: str
    demonstration_examples: List
    best_prompt: str
    expert_identity: str
    generated_knowledge: str
    task_name: str
