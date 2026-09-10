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
                prompt_format = "Twoje zadanie to udzielenie odpowiedzi na test medyczny dla lekarzy. Spośród odpowiedzi wybierz prawidłową, zwróć tylko i wyłącznie ją. Zakończ odpowiedź kropką.",
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
                prompt_format = "Twoje zadanie to udzielenie odpowiedzi na test medyczny dla lekarzy. Spośród wszystkich odpowiedzi A,B,C,D,E wybierz tylko jedną. Jeżeli nie jesteś pewien, wybierz najbardziej prawdopodobną. Odpowiedz w sposób:\nPrawidłowa odpowiedź to B."
                )
        evaluator.make_predictions()

        #open ended generation
        evaluator = EvaluatorLLM(
                model_id=model_id,
                dataset_str=dataset_str,
                evaluation_methodology = "free_form_generation",
                possible_answers=["A","B","C","D","E"],
                split_name="free_form",
                prompt_format="Twoje zadanie to udzielenie dokładnej i specjalistycznej odpowiedzi na pytanie medyczne dla lekarzy. Odpowiedz jednym zdaniem, frazą, terminem medycznym, najbardziej odpowiednim, poprawnym tokiem postępowania lub diagnozą."
                )
        evaluator.model_init()
        evaluator.parameters_init()
        evaluator.create_messages()
        evaluator.make_predictions()

        # # #Multichoice 1
        evaluator.prompt_init(
                split_name = "multiple_choice",
                evaluation_methodology = "our_method",
                prompt_format = "Twoje zadanie to udzielenie odpowiedzi na test medyczny dla lekarzy. Spośród stwierdzeń wybierz wszystkie które są prawdziwe, zwróć tylko i wyłącznie ich numery. Zakończ odpowiedź kropką.",
                possible_answers = [str(x) for x in range(20)]
                )
        evaluator.make_predictions()

        #Multichoice 2
        evaluator.prompt_init(
                split_name = "multiple_choice2",
                evaluation_methodology = "our_method",
                prompt_format = "Twoje zadanie to udzielenie odpowiedzi na test medyczny dla lekarzy. Spośród odpowiedzi wybierz wszystkie które są poprawne, zwróć tylko i wyłącznie je. Zakończ odpowiedź kropką.",
                possible_answers = ["A","B","C","D","E"]
                )
        evaluator.make_predictions()

        # #Abstaining substitution
        evaluator.prompt_init(
                split_name = "abstaining_substitution",
                evaluation_methodology = "our_method",
                prompt_format = "Twoje zadanie to udzielenie odpowiedzi na test medyczny dla lekarzy. Spośród odpowiedzi wybierz prawidłową, zwróć tylko i wyłącznie ją. Zakończ odpowiedź kropką.",
                possible_answers = ["A","B","C","D","E"]
                )
        evaluator.make_predictions()
