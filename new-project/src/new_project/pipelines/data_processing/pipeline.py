"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from new_project.pipelines.data_processing.nodes import preprocess_application_train, preprocess_installments_payments, preprocess_previous_application, preprocess_credit_card_balance, preprocess_bureau, preprocess_bureau_balance, preprocess_pos_cash_balance,create_model_input_table

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_application_train,
                inputs="application_train",
                outputs="preprocessed_application_train",
                name="preprocess_application_train_node",
            ),
            node(
                func=preprocess_installments_payments,
                inputs="installments_payments",
                outputs="preprocessed_installments_payments",
                name="preprocess_installments_payments_node",
            ),
             node(
                func=preprocess_previous_application,
                inputs="previous_application",
                outputs="preprocessed_previous_application",
                name="preprocess_previous_application_node",
            ),
            node(
                func=preprocess_credit_card_balance,
                inputs="credit_card_balance",
                outputs="preprocessed_credit_card_balance",
                name="preprocess_credit_card_balance_node",
            ),
             node(
                func=preprocess_bureau,
                inputs="bureau",
                outputs="preprocessed_bureau",
                name="preprocess_bureau_node",
            ),
            node(
                func=preprocess_bureau_balance,
                inputs="bureau_balance",
                outputs="preprocessed_bureau_balance",
                name="preprocess_bureau_balance_node",
            ),
            node(
                func=preprocess_pos_cash_balance,
                inputs="pos_cash_balance",
                outputs="preprocessed_pos_cash_balance",
                name="preprocess_pos_cash_balance_node",
            ),
            node(
                func=create_model_input_table,
                inputs=['preprocessed_pos_cash_balance', 'preprocessed_bureau_balance', 'preprocessed_bureau', 'preprocessed_credit_card_balance', 'preprocessed_previous_application', 'preprocessed_installments_payments', 'preprocessed_application_train'],
                outputs="model_input_table",
                name="create_model_input_table_node",
),
        ]
   )