"""
    Testing the idea of refinfining membership by constructing members based on increasing "distance" from true members.
    Try both actual edit distance, and neighbors generated by the MIA attack (more semantically similar)
"""
import numpy as np
import torch
from tqdm import tqdm
import random
import datetime
import os
import json
import pickle
import math
from collections import defaultdict
from functools import partial
from typing import List, Dict

from simple_parsing import ArgumentParser
from pathlib import Path

from mimir.config import (
    ExperimentConfig,
    EnvironmentConfig,
    NeighborhoodConfig,
    ReferenceConfig,
    OpenAIConfig,
)
import mimir.data_utils as data_utils
import mimir.plot_utils as plot_utils
from mimir.utils import fix_seed
from mimir.models import LanguageModel, ReferenceModel
from mimir.attacks.blackbox_attacks import BlackBoxAttacks, Attack
from mimir.attacks.neighborhood import T5Model, BertModel, NeighborhoodAttack
from mimir.attacks.utils import get_attacker

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def get_attackers(
    target_model,
    ref_models,
    config: ExperimentConfig,
):
    # Look at all attacks, and attacks that we have implemented
    attacks = config.blackbox_attacks
    implemented_blackbox_attacks = [a.value for a in BlackBoxAttacks]
    # check for unimplemented attacks
    runnable_attacks = []
    for a in attacks:
        if a not in implemented_blackbox_attacks:
            print(f"Attack {a} not implemented, will be ignored")
            pass
        runnable_attacks.append(a)
    attacks = runnable_attacks

    # Initialize attackers
    attackers = {}
    for attack in attacks:
        if attack != BlackBoxAttacks.REFERENCE_BASED:
            attackers[attack] = get_attacker(attack)(config, target_model)

    # Initialize reference-based attackers if specified
    if ref_models is not None:
        for name, ref_model in ref_models.items():
            attacker = get_attacker(BlackBoxAttacks.REFERENCE_BASED)(
                config, target_model, ref_model
            )
            attackers[f"{BlackBoxAttacks.REFERENCE_BASED}-{name.split('/')[-1]}"] = attacker
    return attackers


def get_mia_scores(
    data,
    attackers_dict: Dict[str, Attack],
    target_model: LanguageModel,
    ref_models: Dict[str, ReferenceModel],
    config: ExperimentConfig,
    n_samples: int = None,
    batch_size: int = 50,
):
    # Fix randomness
    fix_seed(config.random_seed)

    n_samples = len(data["records"]) if n_samples is None else n_samples

    # Look at all attacks, and attacks that we have implemented
    neigh_config = config.neighborhood_config

    if neigh_config:
        n_perturbation_list = neigh_config.n_perturbation_list
        in_place_swap = neigh_config.original_tokenization_swap

    results = []
    neighbors = None
    if BlackBoxAttacks.NEIGHBOR in attackers_dict.keys() and neigh_config.load_from_cache:
        neighbors = data[f"neighbors"]
        print("Loaded neighbors from cache!")

    # For each batch of data
    # TODO: Batch-size isn't really "batching" data - change later
    for batch in tqdm(range(math.ceil(n_samples / batch_size)), desc=f"Computing criterion"):
        texts = data["records"][batch * batch_size : (batch + 1) * batch_size]

        # For each entry in batch
        for idx in range(len(texts)):
            sample_information = defaultdict(list)
            sample = (
                texts[idx][: config.max_substrs]
                if config.full_doc
                else [texts[idx]]
            )

            # This will be a list of integers if pretokenized
            sample_information["sample"] = sample
            if config.pretokenized:
                detokenized_sample = [target_model.tokenizer.decode(s) for s in sample]
                sample_information["detokenized"] = detokenized_sample

            # For each substring
            for i, substr in enumerate(sample):
                # compute token probabilities for sample
                s_tk_probs = (
                    target_model.get_probabilities(substr)
                    if not config.pretokenized
                    else target_model.get_probabilities(
                        detokenized_sample[i], tokens=substr
                    )
                )

                # Always compute LOSS score. Also helpful for reference-based and many other attacks.
                loss = (
                    target_model.get_ll(substr, probs=s_tk_probs)
                    if not config.pretokenized
                    else target_model.get_ll(
                        detokenized_sample[i], tokens=substr, probs=s_tk_probs
                    )
                )
                sample_information[BlackBoxAttacks.LOSS].append(loss)

                # TODO: Shift functionality into each attack entirely, so that this is just a for loop
                # For each attack
                for attack, attacker in attackers_dict.items():
                    # LOSS already added above, Reference handled later
                    if attack.startswith(BlackBoxAttacks.REFERENCE_BASED) or attack == BlackBoxAttacks.LOSS:
                        continue

                    if attack != BlackBoxAttacks.NEIGHBOR:
                        score = attacker.attack(
                            substr,
                            probs=s_tk_probs,
                            detokenized_sample=(
                                detokenized_sample[i]
                                if config.pretokenized
                                else None
                            ),
                            loss=loss,
                        )
                        sample_information[attack].append(score)
                    else:
                        # For each 'number of neighbors'
                        for n_perturbation in n_perturbation_list:
                            # Only run if neighbors available available
                            if neighbors and not neigh_config.dump_cache:
                                substr_neighbors = neighbors[n_perturbation][
                                    batch * batch_size + idx
                                ][i]

                                # Only evaluate neighborhood attack when not caching neighbors
                                score = attacker.attack(
                                    substr,
                                    probs=s_tk_probs,
                                    detokenized_sample=(
                                        detokenized_sample[i]
                                        if config.pretokenized
                                        else None
                                    ),
                                    loss=loss,
                                    batch_size=4,
                                    substr_neighbors=substr_neighbors,
                                )

                                sample_information[
                                    f"{attack}-{n_perturbation}"
                                ].append(score)

            # Add the scores we collected for each sample for each
            # attack into to respective list for its classification
            results.append(sample_information)

    # Perform reference-based attacks
    if ref_models is not None:
        for name, _ in ref_models.items():
            ref_key = f"{BlackBoxAttacks.REFERENCE_BASED}-{name.split('/')[-1]}"
            attacker = attackers_dict.get(ref_key, None)
            if attacker is None:
                continue

            # Update collected scores for each sample with ref-based attack scores
            for r in tqdm(results, desc="Ref scores"):
                ref_model_scores = []
                for i, s in enumerate(r["sample"]):
                    if config.pretokenized:
                        s = r["detokenized"][i]
                    score = attacker.attack(s, probs=None,
                                                loss=r[BlackBoxAttacks.LOSS][i])
                    ref_model_scores.append(score)
                r[ref_key].extend(ref_model_scores)

            attacker.unload()
    else:
        print("No reference models specified, skipping Reference-based attacks")

    # Rearrange the nesting of the results dict and calculated aggregated score for sample
    # attack -> member/nonmember -> list of scores
    samples = []
    predictions = defaultdict(lambda: [])
    for r in results:
        samples.append(r["sample"])
        for attack, scores in r.items():
            if attack != "sample" and attack != "detokenized":
                # TODO: Is there a reason for the np.min here?
                predictions[attack].append(np.min(scores))

    return predictions, samples


def generate_data(
    dataset: str,
    train: bool = True,
    presampled: str = None,
    specific_source: str = None,
    mask_model_tokenizer = None
):
    data_obj = data_utils.Data(dataset, config=config, presampled=presampled)
    data = data_obj.load(
        train=train,
        mask_tokenizer=mask_model_tokenizer,
        specific_source=specific_source,
    )
    return data_obj, data
    # return generate_samples(data[:n_samples], batch_size=batch_size)


if __name__ == "__main__":
    # TODO: Shift below to main() function - variables here are global and may interfe with functions etc.

    # Extract relevant configurations from config file
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--config", help="Path to attack config file", type=Path)
    args, remaining_argv = parser.parse_known_args()
    # Attempt to extract as much information from config file as you can
    config = ExperimentConfig.load(args.config, drop_extra_fields=False)
    # Also give user the option to provide config values over CLI
    parser = ArgumentParser(parents=[parser])
    parser.add_arguments(ExperimentConfig, dest="exp_config", default=config)
    args = parser.parse_args(remaining_argv)
    config: ExperimentConfig = args.exp_config

    env_config: EnvironmentConfig = config.env_config
    neigh_config: NeighborhoodConfig = config.neighborhood_config
    ref_config: ReferenceConfig = config.ref_config
    openai_config: OpenAIConfig = config.openai_config

    if neigh_config:
        if neigh_config.load_from_cache and neigh_config.dump_cache:
            raise ValueError(
                "Cannot dump and load from cache at the same time. Please set one of these to False"
            )

    # generic generative model
    base_model = LanguageModel(config)

    # reference model if we are doing the ref-based attack
    ref_models = None
    if (
        ref_config is not None
        and BlackBoxAttacks.REFERENCE_BASED in config.blackbox_attacks
    ):
        ref_models = {
            model: ReferenceModel(config, model) for model in ref_config.models
        }

    # Prepare attackers
    attackers_dict = get_attackers(base_model, ref_models, config)

    # Load neighborhood attack model, only if we are doing the neighborhood attack AND generating neighbors
    mask_model = None
    if (
        neigh_config
        and (not neigh_config.load_from_cache)
        and (BlackBoxAttacks.NEIGHBOR in config.blackbox_attacks)
    ):
        attacker_ne = attackers_dict[BlackBoxAttacks.NEIGHBOR]
        mask_model = attacker_ne.get_mask_model()

    print("MOVING BASE MODEL TO GPU...", end="", flush=True)
    base_model.load()

    print(f"Loading dataset {config.dataset_member} and {config.dataset_nonmember}...")
    # data, seq_lens, n_samples = generate_data(config.dataset_member)

    data_obj_mem, data_member = generate_data(
        config.dataset_member,
        presampled=config.presampled_dataset_member,
        mask_model_tokenizer=mask_model.tokenizer if mask_model else None
    )

    ### <LOGIC FOR SPECIFIC EXPERIMENTS>
    def edit(x, n: int):
        """
        Return version of x that has some distance 'n' from x.
        Could be edit-distance based, or semantic distance (NE) based
        """
        # Tokenize sentence
        x_tok = base_model.tokenizer(x)["input_ids"]
        # Pick n random positions
        positions = np.random.choice(len(x_tok), n, replace=False)
        # Replace those positions with random words from vocabulary
        for pos in positions:
            x_tok[pos] = np.random.choice(base_model.tokenizer.vocab_size)
        # Detokenize
        x = base_model.tokenizer.decode(x_tok)
        return x

    if config.load_from_cache and not config.dump_cache:
        # For NE neighbors, 30% is masked
        with open(
            f"/mmfs1/gscratch/h2lab/micdun/mimir/data/gpt_generated_paraphrases/out/em_version_{config.specific_source}_paraphrases_1000_samples_5_trials.jsonl", #f"edit_distance_members/ne/{config.specific_source}.json",
            "r",
        ) as f:
            other_members_data = json.load(f)
            n_try = list(other_members_data.keys())
            n_trials = len(other_members_data[n_try[0]])
    elif config.dump_cache:
        # Try out multiple "distances"
        n_try = [1, 5, 10, 25, 100]
        # With multiple trials
        n_trials = 50
        other_members_data = {}
        for n in tqdm(n_try, "Generating edited members"):
            trials = {}
            for i in tqdm(range(n_trials)):
                trials[i] = [edit(x, n) for x in data_member]
            other_members_data[n] = trials
        with open(
            f"edit_distance_members/ne/{config.specific_source}.json",
            "w",
        ) as f:
            json.dump(other_members_data, f)
        print("Data dumped! Please re-run with load_from_cache set to True")
        exit(0)
    ### </LOGIC FOR SPECIFIC EXPERIMENTS>
    score_dict = defaultdict(lambda: defaultdict(dict))

    pbar = tqdm(total=len(n_try) * n_trials)
    for n, other_member in other_members_data.items():
        ds_objects = {"member": data_obj_mem}
        for i in range(n_trials):
            n_samples = len(other_member[str(i)])
            other_blackbox_predictions, _ = get_mia_scores(
                data={"records": other_member[str(i)]},
                attackers_dict=attackers_dict,
                target_model=base_model,
                ref_models=ref_models,
                config=config,
                n_samples=n_samples
            )
            pbar.update(1)

            for attack in other_blackbox_predictions.keys():
                score_dict[attack][n][i] = other_blackbox_predictions[attack]

    pbar.close()
    with open(f"edit_distance_members/scores/gpt_paraphrase_results_{config.specific_source}.json", "w") as f:
        json.dump(score_dict, f, indent=4)
