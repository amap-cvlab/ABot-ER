class OutputDataAfterEngine:
    def __init__(
        self,
        predict,
        generated_tokens_with_probs=None,
        predict_finish_reasons=None,
        predict_response_lengths=None,
        request_ids=None,
        audio_path=None,
        input_generate_token_cnt=None,
        output_generate_token_cnt=None,
    ):
        self.predict = predict
        self.generated_tokens_with_probs = generated_tokens_with_probs
        self.predict_finish_reasons = predict_finish_reasons
        self.predict_response_lengths = predict_response_lengths
        self.request_ids = request_ids
        self.audio_path = audio_path
        self.input_generate_token_cnt = input_generate_token_cnt
        self.output_generate_token_cnt = output_generate_token_cnt
