from kedro.pipeline import Pipeline, node

from .nodes import load_merged_ontology, refine_ontology


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_merged_ontology,
                inputs="params:ontology_refinement.input_path",
                outputs="merged_ontology_for_refinement",
                name="load_merged_ontology_node",
            ),
            node(
                func=refine_ontology,
                inputs=[
                    "merged_ontology_for_refinement",
                    "params:ontology_refinement.wikidata",
                    "params:ontology_refinement.vocabulary_path",
                ],
                # both are catalog datasets (conf/base/catalog.yml)
                outputs=["refined_ontology", "refinement_report"],
                name="refine_ontology_node",
            ),
        ]
    )
