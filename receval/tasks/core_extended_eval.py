import csv
import json
import time
import random

from receval.tasks.core_eval import (
    print0,
    download_eval_bundle,
    evaluate_task,
)

CORE_TASK_LABELS = {
    "hellaswag_zeroshot", "jeopardy", "bigbench_qa_wikidata", "arc_easy", "arc_challenge",
    "copa", "commonsense_qa", "piqa", "openbook_qa", "lambada_openai", "hellaswag",
    "winograd", "winogrande", "bigbench_dyck_languages", "agi_eval_lsat_ar",
    "bigbench_cs_algorithms", "bigbench_operators", "bigbench_repeat_copy_logic",
    "squad", "coqa", "boolq", "bigbench_language_identification",
}

CORE_EXTENDED_TASKS = [
    {"label": "hellaswag_zeroshot", "dataset_uri": "language_understanding/hellaswag.jsonl", "num_fewshot": [0], "icl_task_type": "multiple_choice"},
    {"label": "jeopardy", "dataset_uri": "world_knowledge/jeopardy_all.jsonl", "num_fewshot": [10], "icl_task_type": "language_modeling", "continuation_delimiter": "\nAnswer: "},
    {"label": "bigbench_qa_wikidata", "dataset_uri": "world_knowledge/bigbench_qa_wikidata.jsonl", "num_fewshot": [10], "icl_task_type": "language_modeling"},
    {"label": "arc_easy", "dataset_uri": "world_knowledge/arc_easy.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice", "continuation_delimiter": "\nAnswer: "},
    {"label": "arc_challenge", "dataset_uri": "world_knowledge/arc_challenge.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice", "continuation_delimiter": "\nAnswer: "},
    {"label": "copa", "dataset_uri": "commonsense_reasoning/copa.jsonl", "num_fewshot": [0], "icl_task_type": "multiple_choice"},
    {"label": "commonsense_qa", "dataset_uri": "commonsense_reasoning/commonsense_qa.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice"},
    {"label": "piqa", "dataset_uri": "commonsense_reasoning/piqa.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice", "continuation_delimiter": "\nAnswer: "},
    {"label": "openbook_qa", "dataset_uri": "commonsense_reasoning/openbook_qa.jsonl", "num_fewshot": [0], "icl_task_type": "multiple_choice"},
    {"label": "lambada_openai", "dataset_uri": "language_understanding/lambada_openai.jsonl", "num_fewshot": [0], "icl_task_type": "language_modeling"},
    {"label": "hellaswag", "dataset_uri": "language_understanding/hellaswag.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice"},
    {"label": "winograd", "dataset_uri": "language_understanding/winograd_wsc.jsonl", "num_fewshot": [0], "icl_task_type": "schema"},
    {"label": "winogrande", "dataset_uri": "language_understanding/winogrande.jsonl", "num_fewshot": [0], "icl_task_type": "schema"},
    {"label": "bigbench_dyck_languages", "dataset_uri": "symbolic_problem_solving/bigbench_dyck_languages.jsonl", "num_fewshot": [10], "icl_task_type": "language_modeling"},
    {"label": "agi_eval_lsat_ar", "dataset_uri": "symbolic_problem_solving/agi_eval_lsat_ar.jsonl", "num_fewshot": [3], "icl_task_type": "multiple_choice"},
    {"label": "bigbench_cs_algorithms", "dataset_uri": "symbolic_problem_solving/bigbench_cs_algorithms.jsonl", "num_fewshot": [10], "icl_task_type": "language_modeling"},
    {"label": "bigbench_operators", "dataset_uri": "symbolic_problem_solving/bigbench_operators.jsonl", "num_fewshot": [10], "icl_task_type": "language_modeling"},
    {"label": "bigbench_repeat_copy_logic", "dataset_uri": "symbolic_problem_solving/bigbench_repeat_copy_logic.jsonl", "num_fewshot": [10], "icl_task_type": "language_modeling"},
    {"label": "squad", "dataset_uri": "reading_comprehension/squad.jsonl", "num_fewshot": [10], "icl_task_type": "language_modeling"},
    {"label": "coqa", "dataset_uri": "reading_comprehension/coqa.jsonl", "num_fewshot": [0], "icl_task_type": "language_modeling"},
    {"label": "boolq", "dataset_uri": "reading_comprehension/boolq.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice", "continuation_delimiter": "\nAnswer: "},
    {"label": "bigbench_language_identification", "dataset_uri": "language_understanding/bigbench_language_identification.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice"},
    {"label": "mmlu_zeroshot", "dataset_uri": "world_knowledge/mmlu.jsonl", "num_fewshot": [0], "icl_task_type": "multiple_choice", "continuation_delimiter": "Answer: "},
    {"label": "mmlu_fewshot", "dataset_uri": "world_knowledge/mmlu.jsonl", "num_fewshot": [5], "icl_task_type": "multiple_choice", "continuation_delimiter": "\nAnswer: "},
    {"label": "siqa", "dataset_uri": "commonsense_reasoning/siqa.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice"},
    {"label": "bigbench_misconceptions", "dataset_uri": "world_knowledge/bigbench_misconceptions.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice"},
    {"label": "bigbench_novel_concepts", "dataset_uri": "commonsense_reasoning/bigbench_novel_concepts.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice"},
    {"label": "bigbench_strange_stories", "dataset_uri": "commonsense_reasoning/bigbench_strange_stories.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice"},
    {"label": "bigbench_strategy_qa", "dataset_uri": "commonsense_reasoning/bigbench_strategy_qa.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice"},
    {"label": "bigbench_conlang_translation", "dataset_uri": "language_understanding/bigbench_conlang_translation.jsonl", "num_fewshot": [0], "icl_task_type": "language_modeling"},
    {"label": "bigbench_conceptual_combinations", "dataset_uri": "language_understanding/bigbench_conceptual_combinations.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice"},
    {"label": "bigbench_elementary_math_qa", "dataset_uri": "symbolic_problem_solving/bigbench_elementary_math_qa.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice"},
    {"label": "bigbench_logical_deduction", "dataset_uri": "symbolic_problem_solving/bigbench_logical_deduction.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice"},
    {"label": "simple_arithmetic_nospaces", "dataset_uri": "symbolic_problem_solving/simple_arithmetic_nospaces.jsonl", "num_fewshot": [10], "icl_task_type": "language_modeling"},
    {"label": "simple_arithmetic_withspaces", "dataset_uri": "symbolic_problem_solving/simple_arithmetic_withspaces.jsonl", "num_fewshot": [10], "icl_task_type": "language_modeling"},
    {"label": "math_qa", "dataset_uri": "symbolic_problem_solving/math_qa.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice"},
    {"label": "logi_qa", "dataset_uri": "symbolic_problem_solving/logi_qa.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice", "continuation_delimiter": "\nAnswer: "},
    {"label": "pubmed_qa_labeled", "dataset_uri": "reading_comprehension/pubmed_qa_labeled.jsonl", "num_fewshot": [10], "icl_task_type": "language_modeling"},
    {"label": "agi_eval_lsat_rc", "dataset_uri": "reading_comprehension/agi_eval_lsat_rc.jsonl", "num_fewshot": [3], "icl_task_type": "multiple_choice"},
    {"label": "agi_eval_lsat_lr", "dataset_uri": "reading_comprehension/agi_eval_lsat_lr.jsonl", "num_fewshot": [3], "icl_task_type": "multiple_choice"},
    {"label": "bigbench_understanding_fables", "dataset_uri": "reading_comprehension/bigbench_understanding_fables.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice"},
    {"label": "agi_eval_sat_en", "dataset_uri": "reading_comprehension/agi_eval_sat_en.jsonl", "num_fewshot": [3], "icl_task_type": "multiple_choice"},
    {"label": "winogender_mc_female", "dataset_uri": "safety/winogender_mc_female.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice"},
    {"label": "winogender_mc_male", "dataset_uri": "safety/winogender_mc_male.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice"},
    {"label": "enterprise_pii_classification", "dataset_uri": "safety/enterprise_pii_classification.jsonl", "num_fewshot": [10], "icl_task_type": "multiple_choice"},
    {"label": "bbq", "dataset_uri": "safety/bbq.jsonl", "num_fewshot": [3], "icl_task_type": "multiple_choice"},
]

def run_core_extended_eval(model, tokenizer, device, max_seq_len=None, max_per_task=-1, seeds=None):
    if seeds is None:
        seeds = [1234]
    eval_bundle = download_eval_bundle()
    data_path = eval_bundle / "eval_data"
    meta_path = eval_bundle / "eval_meta_data.csv"
    baselines = {}
    with open(meta_path, 'r') as f:
        for row in csv.DictReader(f):
            baselines[row['Eval Task']] = float(row['Random baseline'])
    all_seed_results = {}
    for seed in seeds:
        if len(seeds) > 1:
            print0(f"\n{'='*60}")
            print0(f"Running CORE Extended eval with seed={seed}")
            print0(f"{'='*60}")
        results, centered = {}, {}
        for task in CORE_EXTENDED_TASKS:
            label = task['label']
            task_meta = {
                'task_type': task['icl_task_type'],
                'num_fewshot': task['num_fewshot'][0],
                'continuation_delimiter': task.get('continuation_delimiter', ' ')
            }
            dataset_path = data_path / task['dataset_uri']
            if not dataset_path.exists():
                print0(f"Skipping {label}: dataset not found")
                continue
            print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')
            t0 = time.time()
            with open(dataset_path, 'r') as f:
                data = [json.loads(line) for line in f]
            rng = random.Random(1337)
            rng.shuffle(data)
            if max_per_task > 0:
                data = data[:max_per_task]
            acc = evaluate_task(model, tokenizer, data, device, task_meta, max_seq_len, seed=seed)
            results[label] = acc
            baseline = baselines.get(label, 25.0)
            centered[label] = (acc - 0.01 * baseline) / (1.0 - 0.01 * baseline)
            print0(f"accuracy: {acc:.4f} | centered: {centered[label]:.4f} | time: {time.time()-t0:.2f}s")
        core_centered = {k: v for k, v in centered.items() if k in CORE_TASK_LABELS}
        core_score = sum(core_centered.values()) / len(core_centered) if core_centered else 0.0
        core_extended_score = sum(centered.values()) / len(centered) if centered else 0.0
        all_seed_results[seed] = {
            'results': results, 'centered_results': centered,
            'core_metric': core_score, 'core_extended_metric': core_extended_score
        }
        if len(seeds) > 1:
            print0(f"\nSeed {seed} CORE: {core_score:.4f} | CORE Extended: {core_extended_score:.4f}")
    if len(seeds) == 1:
        return all_seed_results[seeds[0]]
    task_labels = list(all_seed_results[seeds[0]]['results'].keys())
    avg_results, std_results, avg_centered, std_centered = {}, {}, {}, {}
    for label in task_labels:
        accs = [all_seed_results[s]['results'][label] for s in seeds]
        cents = [all_seed_results[s]['centered_results'][label] for s in seeds]
        avg_results[label] = sum(accs) / len(accs)
        avg_centered[label] = sum(cents) / len(cents)
        std_results[label] = (sum((x - avg_results[label])**2 for x in accs) / len(accs)) ** 0.5
        std_centered[label] = (sum((x - avg_centered[label])**2 for x in cents) / len(cents)) ** 0.5
    core_scores = [all_seed_results[s]['core_metric'] for s in seeds]
    ext_scores = [all_seed_results[s]['core_extended_metric'] for s in seeds]
    avg_core = sum(core_scores) / len(core_scores)
    std_core = (sum((x - avg_core)**2 for x in core_scores) / len(core_scores)) ** 0.5
    avg_ext = sum(ext_scores) / len(ext_scores)
    std_ext = (sum((x - avg_ext)**2 for x in ext_scores) / len(ext_scores)) ** 0.5
    print0(f"\n{'='*60}")
    print0(f"AGGREGATED RESULTS (across {len(seeds)} seeds)")
    print0(f"{'='*60}")
    for label in task_labels:
        print0(f"  {label}: acc={avg_results[label]:.4f}±{std_results[label]:.4f} "
               f"centered={avg_centered[label]:.4f}±{std_centered[label]:.4f}")
    print0(f"\n  CORE metric: {avg_core:.4f} ± {std_core:.4f}")
    print0(f"  CORE Extended metric: {avg_ext:.4f} ± {std_ext:.4f}")
    return {
        'per_seed': all_seed_results,
        'aggregated': {
            'results': avg_results, 'results_std': std_results,
            'centered_results': avg_centered, 'centered_results_std': std_centered,
            'core_metric': avg_core, 'core_metric_std': std_core,
            'core_extended_metric': avg_ext, 'core_extended_metric_std': std_ext,
        },
        'seeds': seeds,
        'results': avg_results,
        'centered_results': avg_centered,
        'core_metric': avg_core,
        'core_extended_metric': avg_ext,
    }
