from taco_shell_rag.core.contracts import RerankerInfo, RerankResult
from taco_shell_rag.model_config import RerankerOptions
from taco_shell_rag.model_runtime import check_loaded_model, require_cuda


class CrossEncoderReranker:
    def __init__(self, options: RerankerOptions):
        import torch
        from sentence_transformers import CrossEncoder

        require_cuda(torch)
        self.options = options
        self.model = CrossEncoder(
            options.model,
            revision=options.revision,
            device=options.device,
            backend="torch",
            model_kwargs={"dtype": getattr(torch, options.precision)},
            processor_kwargs={"model_max_length": options.max_length},
            trust_remote_code=False,
            activation_fn=torch.nn.Identity(),
        )
        check_loaded_model(self.model, options.precision)
        if self.model.num_labels != 1:
            raise ValueError("reranking requires one scalar per query/document pair")
        if self.model.max_seq_length != options.max_length:
            raise ValueError("the requested reranker max_length was not applied")
        prompt = options.prompt
        if prompt is None:
            prompt = self.model.prompts.get(self.model.default_prompt_name)
        self.info = RerankerInfo(
            options.model,
            options.revision,
            str(self.model.device),
            options.backend,
            options.precision,
            self.model.max_seq_length,
            options.batch_size,
            prompt,
        )

    def describe(self):
        return self.info

    def rerank(self, query, texts):
        scores = self.model.predict(
            [(query, text) for text in texts],
            batch_size=self.options.batch_size,
            prompt=self.options.prompt,
            show_progress_bar=False,
        ).tolist()
        return RerankResult(scores, self.info)
