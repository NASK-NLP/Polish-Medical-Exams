import sys
from scripts.evaluation.evaluator import EvaluatorLLM
dataset_str  = sys.argv[1]
model_id = sys.argv[2]
if __name__ == '__main__':
        # evaluation GT our_method
        evaluator = EvaluatorLLM(
                model_id=model_id,
                dataset_str=dataset_str,
                evaluation_methodology = "our_method",
                possible_answers=["A","B","C","D","E"],
                split_name="our_method",
                prompt_format = "Your task is to provide answers to a medical test for doctors. From the provided answers, select the correct one and respond only with that answer. End your answer with a period.",
                extension="_pandas.pickle")

        evaluator.model_init()
        evaluator.parameters_init()
        evaluator.create_messages()
        evaluator.make_predictions()
        
        # evaluation GT prev_method
        evaluator.prompt_init(
                evaluation_methodology = "prev_method",
                possible_answers=["A","B","C","D","E"],
                split_name="prev_method",
                prompt_format = "Your task is to provide answers to a medical test for doctors. From all the provided answers A, B, C, D, E select only one. If you are not sure, choose the most probable one. Answer in a manner:\nCorrect answer is B."
                )
        evaluator.make_predictions()

        #open ended generation
        evaluator = EvaluatorLLM(
                model_id=model_id,
                dataset_str=dataset_str,
                evaluation_methodology = "free_form_generation",
                possible_answers=["A","B","C","D","E"],
                split_name="free_form",
                prompt_format="Your task is to provide a precise and specialized answer to a medical question for doctors. Answer with one sentence, phrase, medical term, most appropriate clinical course of action, or diagnosis."
                )
        evaluator.model_init()
        evaluator.parameters_init()
        evaluator.create_messages()
        evaluator.make_predictions()

        # # #Multichoice 1
        evaluator.prompt_init(
                split_name = "multiple_choice",
                evaluation_methodology = "our_method",
                prompt_format = "Your task is to provide answers to a medical test for doctors. From the provided statements, select all that are correct and return only their indexes. End your answer with a period.",
                possible_answers = [str(x) for x in range(20)]
                )
        evaluator.make_predictions()

        #Multichoice 2
        evaluator.prompt_init(
                split_name = "multiple_choice2",
                evaluation_methodology = "our_method",
                prompt_format = "Your task is to provide answers to a medical test for doctors. From the provided options, select all the correct ones and respond only with those answers. End your answer with a period.",
                possible_answers = ["A","B","C","D","E"]
                )
        evaluator.make_predictions()

        # #Abstaining substitution
        evaluator.prompt_init(
                split_name = "abstaining_substitution",
                evaluation_methodology = "our_method",
                prompt_format = "Your task is to provide answers to a medical test for doctors. From the provided answers, select the correct one and respond only with that answer. End your answer with a period.",
                possible_answers = ["A","B","C","D","E"]
                )
        evaluator.make_predictions()
