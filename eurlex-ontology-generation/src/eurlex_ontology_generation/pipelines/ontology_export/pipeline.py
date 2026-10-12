from kedro.pipeline import Pipeline, node

from .nodes import export_and_validate, persist_export, quality_gate


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=export_and_validate,
                inputs=[
                    "refined_ontology",                 # catalog dataset
                    "params:ontology_export.base_iri",
                    "params:ontology_export.shapes_path",
                    "params:ontology_export.vocabulary_path",
                    "params:ontology_export.fail_on_violation",
                    "params:ontology_export.run_reasoner",
                    "params:ontology_export.act_metadata_path",
                ],
                outputs=["ontology_rdf_graph", "shacl_report_graph", "ontology_export_summary"],
                name="export_and_validate_node",
            ),
            node(
                func=persist_export,
                inputs=[
                    "ontology_rdf_graph",
                    "shacl_report_graph",
                    "ontology_export_summary",
                    "params:ontology_export.turtle_path",
                    "params:ontology_export.shacl_report_path",
                    "params:ontology_export.summary_path",
                ],
                outputs="ontology_turtle_path",
                name="persist_export_node",
            ),
            node(
                func=quality_gate,
                inputs=["ontology_export_summary", "ontology_turtle_path"],
                outputs="ontology_quality_gate",
                name="quality_gate_node",
            ),
        ]
    )
