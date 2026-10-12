import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils

TOP_K = 5


def softmax_topk(logits, k=TOP_K):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    idx = np.argsort(-p, axis=-1)[..., :k]
    return np.take_along_axis(p, idx, axis=-1), idx


class TritonPythonModel:
    def initialize(self, args):
        path = os.path.join(
            args["model_repository"],
            args["model_version"], "classes.json")
        with open(path) as f:
            self.classes = json.load(f)

    def execute(self, requests):
        responses = []
        for request in requests:
            logits = pb_utils.get_input_tensor_by_name(
                request, "logits").as_numpy()
            probs, idx = softmax_topk(
                logits.astype(np.float32))
            labels = np.array(
                [[self.classes[j].encode("utf-8")
                  for j in row] for row in idx],
                dtype=np.object_)
            responses.append(pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "top_probs",
                        probs.astype(np.float32)),
                    pb_utils.Tensor("top_labels", labels),
                ]))
        return responses

    def finalize(self):
        pass
