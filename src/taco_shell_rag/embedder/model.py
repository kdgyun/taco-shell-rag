from taco_shell_rag.core.contracts import EmbeddingResult, EmbeddingSpace
from taco_shell_rag.model_config import EmbeddingOptions
from taco_shell_rag.model_runtime import check_loaded_model, require_cuda


class SentenceTransformerEmbedder:
    def __init__(self, options: EmbeddingOptions):
        import torch
        from sentence_transformers import SentenceTransformer

        require_cuda(torch)
        self.options = options
        self.model = SentenceTransformer(
            options.model,
            revision=options.revision,
            device=options.device,
            backend="torch",
            model_kwargs={"dtype": getattr(torch, options.precision)},
            trust_remote_code=False,
        )
        check_loaded_model(self.model, options.precision)
        if options.max_length is not None:
            if options.max_length > self.model.max_seq_length:
                raise ValueError("embedding max_length exceeds the model's supported length")
            self.model.max_seq_length = options.max_length
        self.info = EmbeddingSpace(
            options.model,
            options.revision,
            self.model.get_embedding_dimension(),
            str(self.model.device),
            {
                "backend": options.backend,
                "precision": options.precision,
                "max_length": self.model.max_seq_length,
                "query_prompt": options.query_prompt,
                "document_prompt": options.document_prompt,
            },
        )

    def space(self):
        return self.info

    def embed(self, texts, purpose):
        method = self.model.encode_query if purpose == "query" else self.model.encode_document
        prompt = self.options.query_prompt if purpose == "query" else self.options.document_prompt
        vectors = method(
            texts,
            batch_size=self.options.batch_size,
            prompt=prompt,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        return EmbeddingResult(vectors, self.info)
