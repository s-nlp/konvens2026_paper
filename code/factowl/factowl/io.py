import json
import numpy as np
import os
import pandas as pd
from nltk import sent_tokenize


def load_simple_json(p):
    df = pd.read_json(p)
    return df.topics.to_list(), df.generations.to_list()


def load_json_generations(p):
    topics = []
    gens = []
    with open(p, 'r', encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            t = doc["topic"]
            g = doc["output"]
            if doc.get("annotations") is not None:
                anns = doc["annotations"]
                drop = True
                for ann in anns:
                    haf = ann.get("human-atomic-facts")
                    if haf is not None and haf != "null":
                        drop = False

            else:
                continue
            if not drop:
                topics.append(t)
                gens.append(g)
    return topics, gens


def load_json_atomic_facts(p):
    topics = []
    gens = []
    with open(p, 'r', encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            t = doc["topic"]
            g = doc["output"]

            topics.append(t)
            gens.append(g)
    return topics, gens


def save_predictions(eval_dict, save_path, print_res=True):
    d = os.path.dirname(save_path)
    if not os.path.exists(d):
        os.makedirs(d)
    decisions = eval_dict["decisions"]
    samples = []
    dedup_samples = []
    seen_atoms = set()
    unique_topics = set()
    unique_topics_w_atoms = set()
    for sample_id, fact_dict in enumerate(decisions):
        if fact_dict is None:
            continue
        seen_atoms.clear()
        atom = fact_dict["atom"]
        t = fact_dict["topic"]
        unique_topics.add(t)
        if atom is not None:
            unique_topics_w_atoms.add(t)

        sup = fact_dict["is_supported"]
        assert sup == np.True_ or sup == np.False_
        label = 1 if sup == np.True_ else 0
        d = {
            "sample_id": sample_id,
            "topic": fact_dict["topic"],
            "atom": fact_dict["atom"],
            "is_supported": fact_dict["is_supported"],
            "label": label,
            "context": json.dumps(fact_dict.get("context", []), ensure_ascii=False),
            "num_context_passages": len(fact_dict.get("context", []))
        }
        if d.get("true_label") is not None:
            d["true_label"] = d["true_label"]
        samples.append(d)
        if atom not in seen_atoms:
            dedup_samples.append(d)

            seen_atoms.add(atom)

    num_facts = len(samples)  # sum(len(x) for x in samples)
    # num_dedup_facts = sum(len(x) for x in dedup_samples)
    if print_res:
        print(f"Mean num facts in topics with at least 1 fact: {num_facts / len(unique_topics_w_atoms)}")
        print(f"Mean num facts in topics (with fact-less topics): {num_facts / len(unique_topics)}")

        # print(f"Mean score: {score}")
        if eval_dict.get('init_score') is not None:
            print(f"init_score: {eval_dict['init_score']}")

        print(f"Method's score: {eval_dict['score']}")
    # eval_dict["mean_custom_score"] = score
    # eval_dict["mean_custom_deduplicated_score"] = dedup_score

    df = pd.DataFrame(samples)
    print(f"Saving atomic facts DataFrame. Size: {df.shape}, Columns: {df.columns}")
    out_dir = os.path.dirname(save_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df.to_csv(save_path, sep='\t', index=False)


def save_eval_results(eval_dict, save_path):
    d = os.path.dirname(save_path)
    if not os.path.exists(d):
        os.makedirs(d)
    with open(save_path, 'w', encoding="utf-8") as f:
        score = eval_dict["score"]
        respond_ratio = eval_dict["respond_ratio"]
        init_score = eval_dict["init_score"]
        num_facts_per_response = eval_dict["num_facts_per_response"]

        # mcdc = eval_dict["mean_custom_deduplicated_score"]
        # mcc = eval_dict["mean_custom_score"]

        f.write(f"score\t{score}\n")
        f.write(f"init_score\t{init_score}\n")
        f.write(f"respond_ratio\t{respond_ratio}\n")
        f.write(f"num_facts_per_response\t{num_facts_per_response}\n")

        # f.write(f"mean_custom_score\t{mcc}\n")
        # f.write(f"mean_custom_deduplicated_score\t{mcdc}\n")


def json_docs2title_short_passages(docs, max_tokens=256):
    topic2psgs = {}
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    for d in docs:
        to = d["topic"]
        psgs = d["passages"]
        passages = []
        for ps_d in psgs:
            ti = ps_d["title"]
            te = ps_d["text"]
            cur_paras = []
            chunk_size = 0
            sentences = set(sent_tokenize(te))

            for sent in sentences:
                tokens = tokenizer(sent)["input_ids"]
                if chunk_size + len(tokens) > max_tokens:
                    text = ' '.join(cur_paras)
                    passages.append({"title": ti.strip(), "text": text})

                    cur_paras.clear()
                    chunk_size = 0
                cur_paras.append(sent)
                chunk_size += len(tokens)
            if len(cur_paras) > 0:
                text = ' '.join(cur_paras)
                passages.append({"title": ti.strip(), "text": text})

        topic2psgs[to] = passages
    return topic2psgs


def json_docs2title_text_passages(docs):
    topic2psgs = {}
    for d in docs:
        to = d["topic"]
        psgs = d["passages"]
        for ps_d in psgs:
            ti = ps_d["title"]
            te = ps_d["text"]
            ps_d["title"] = f"{to}. {ti}"
        topic2psgs[to] = psgs
    return topic2psgs


def load_json_atomic_facts_w_labels(p):
    topics = []
    atomic_facts = []
    labels = []
    with open(p, 'r', encoding="utf-8") as f:
        docs = json.load(f)
        for anns in docs:
            topic_atomic_facts = []
            topic_labels = []
            if anns is not None:
                for an in anns:
                    # print(an)
                    if an is None:
                        afs = None
                    else:
                        afs = []
                        if an.get("human-atomic-facts") is None:
                            continue
                        afs = [d["text"] for d in an["human-atomic-facts"] if d is not None]
                        af_labels = [d["label"] for d in an["human-atomic-facts"] if d is not None]

                        topic_atomic_facts.extend(afs)
                        topic_labels.extend(af_labels)
                    topic = an["topic"]
            assert len(topic_atomic_facts) == len(topic_labels)
            if len(topic_atomic_facts) == 0:
                topic_atomic_facts = None
                topic_labels = None
            topics.append(topic)
            atomic_facts.append(topic_atomic_facts)
            labels.append(topic_labels)
    assert len(topics) == len(atomic_facts) == len(labels)

    return topics, atomic_facts, labels


def load_lora_adapter_cfg(lora_weights_dir: str):
    cfg_p = os.path.join(lora_weights_dir, "adapter_config.json")
    with open(cfg_p, "r") as f:
        lora_cfg = json.load(f)

    return lora_cfg
