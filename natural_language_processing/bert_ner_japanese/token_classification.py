from typing import List, Tuple

import numpy as np


class TokenClassification:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.id2label = {
            0: "O",
            1: "B-人名",
            2: "I-人名",
            3: "B-法人名",
            4: "I-法人名",
            5: "B-政治的組織名",
            6: "I-政治的組織名",
            7: "B-その他の組織名",
            8: "I-その他の組織名",
            9: "B-地名",
            10: "I-地名",
            11: "B-施設名",
            12: "I-施設名",
            13: "B-製品名",
            14: "I-製品名",
            15: "B-イベント名",
            16: "I-イベント名",
        }

    def gather_pre_entities(
        self,
        input_ids: np.ndarray,
        scores: np.ndarray,
        special_tokens_mask: np.ndarray,
    ):
        """Fuse various numpy arrays into dicts with all the information needed for aggregation"""
        pre_entities = []
        for idx, token_scores in enumerate(scores):
            # Filter special_tokens
            if special_tokens_mask[idx]:
                continue

            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            start_ind = None
            end_ind = None
            is_subword = False

            pre_entity = {
                "word": word,
                "scores": token_scores,
                "start": start_ind,
                "end": end_ind,
                "index": idx,
                "is_subword": is_subword,
            }
            pre_entities.append(pre_entity)

        return pre_entities

    def aggregate(self, pre_entities: List[dict], aggregation=True) -> List[dict]:
        entities = []
        for pre_entity in pre_entities:
            entity_idx = pre_entity["scores"].argmax()
            score = pre_entity["scores"][entity_idx]
            entity = {
                "entity": self.id2label[entity_idx],
                "score": float(score),
                "index": pre_entity["index"],
                "word": pre_entity["word"],
                "start": pre_entity["start"],
                "end": pre_entity["end"],
            }
            entities.append(entity)

        if aggregation:
            entities = self.group_entities(entities)

        return entities

    def aggregate_overlapping_entities(self, entities):
        if len(entities) == 0:
            return entities
        entities = sorted(entities, key=lambda x: x["start"])
        aggregated_entities = []
        previous_entity = entities[0]
        for entity in entities:
            if previous_entity["start"] <= entity["start"] < previous_entity["end"]:
                current_length = entity["end"] - entity["start"]
                previous_length = previous_entity["end"] - previous_entity["start"]
                if current_length > previous_length:
                    previous_entity = entity
                elif (
                    current_length == previous_length
                    and entity["score"] > previous_entity["score"]
                ):
                    previous_entity = entity
            else:
                aggregated_entities.append(previous_entity)
                previous_entity = entity
        aggregated_entities.append(previous_entity)

        return aggregated_entities

    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (`dict`): The entities predicted by the pipeline.
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-", 1)[-1]
        scores = np.nanmean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return entity_group

    def get_tag(self, entity_name: str) -> Tuple[str, str]:
        if entity_name.startswith("B-"):
            bi = "B"
            tag = entity_name[2:]
        elif entity_name.startswith("I-"):
            bi = "I"
            tag = entity_name[2:]
        else:
            # It's not in B-, I- format
            # Default to I- for continuation.
            bi = "I"
            tag = entity_name

        return bi, tag

    def group_entities(self, entities: List[dict]) -> List[dict]:
        """
        Find and group together the adjacent tokens with the same entity predicted.

        Args:
            entities (`dict`): The entities predicted by the pipeline.
        """

        entity_groups = []
        entity_group_disagg = []

        for entity in entities:
            if not entity_group_disagg:
                entity_group_disagg.append(entity)
                continue

            # If the current entity is similar and adjacent to the previous entity,
            # append it to the disaggregated entity group
            # The split is meant to account for the "B" and "I" prefixes
            # Shouldn't merge if both entities are B-type
            bi, tag = self.get_tag(entity["entity"])
            last_bi, last_tag = self.get_tag(entity_group_disagg[-1]["entity"])

            if tag == last_tag and bi != "B":
                # Modify subword type to be previous_type
                entity_group_disagg.append(entity)
            else:
                # If the current entity is different from the previous entity
                # aggregate the disaggregated entity group
                entity_groups.append(self.group_sub_entities(entity_group_disagg))
                entity_group_disagg = [entity]
        if entity_group_disagg:
            # it's the last entity, add it to the entity groups
            entity_groups.append(self.group_sub_entities(entity_group_disagg))

        return entity_groups
