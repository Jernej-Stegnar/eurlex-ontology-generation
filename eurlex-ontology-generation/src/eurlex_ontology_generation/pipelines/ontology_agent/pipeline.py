from kedro.pipeline import Pipeline, node

from ..ontology_generation.nodes import select_chunk_batch
from .nodes import agent_extract_batch


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=select_chunk_batch,
                inputs=[
                    "chunk_batch_manifest",
                    "params:ontology_agent.batch_id",
                    "params:ontology_agent.batch_group_size",
                    "chunk_partition_manifest",
                ],
                outputs="agent_chunk_batch",
                name="select_agent_chunk_batch_node",
            ),
            node(
                func=agent_extract_batch,
                inputs=[
                    "agent_chunk_batch",
                    "chunk_partition_manifest",
                    "params:ontology_agent.batch_id",
                    "params:ontology_agent.batch_group_size",
                    "params:ontology_agent",
                    # same prompts and vocabulary as the generation step, by construction
                    "params:ontology_generation.system_prompt",
                    "params:ontology_generation.user_prompt_template",
                    "params:ontology_generation.vocabulary_path",
                ],
                outputs="agent_batch_summary",
                name="agent_extract_batch_node",
            ),
        ]
    )
