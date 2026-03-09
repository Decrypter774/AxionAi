"""
server.py — Flask chatbot server
"""
import sys
print("Starting server...", flush=True)

try:
    import argparse
    import json
    import os
    from pathlib import Path

    print("Basic imports OK", flush=True)

    from flask import Flask, request, jsonify, Response, send_from_directory
    from flask_cors import CORS
    print("Flask imports OK", flush=True)

    import torch
    import torch.nn.functional as F
    print(f"PyTorch OK, CUDA: {torch.cuda.is_available()}", flush=True)

    from model import GPT, ModelConfig
    from tokenizer import BPETokenizer
    from generate import top_k_filter, top_p_filter
    print("Model imports OK", flush=True)

except Exception as e:
    print(f"IMPORT ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

app = Flask(__name__, static_folder="static")
CORS(app)

model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
model_info = {}

def load_model(checkpoint_path):
    global model, tokenizer, model_info
    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}", flush=True)
        sys.exit(1)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("model_config", {})
    cfg = ModelConfig(**cfg_dict)

    tok_path = ckpt.get("tokenizer_path", "checkpoints/tokenizer.json")
    if not os.path.exists(tok_path):
        tok_path = os.path.join(os.path.dirname(checkpoint_path), "tokenizer.json")
    print(f"Loading tokenizer from: {tok_path}", flush=True)

    tokenizer = BPETokenizer.load(tok_path)
    model = GPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    model_info = {
        "params": f"{model.num_parameters()/1e6:.1f}M",
        "device": device,
        "vocab_size": tokenizer.vocab_size_actual,
        "layers": cfg.n_layers,
        "d_model": cfg.d_model,
        "checkpoint": os.path.basename(checkpoint_path),
    }
    print(f"Model ready: {model_info['params']} params on {device}", flush=True)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/info")
def info():
    return jsonify(model_info)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("prompt", "").strip()
    temperature = float(data.get("temperature", 0.8))
    top_k = int(data.get("top_k", 40))
    top_p = float(data.get("top_p", 0.9))
    max_tokens = int(data.get("max_tokens", 200))
    rep_penalty = float(data.get("repetition_penalty", 1.3))

    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    full_prompt = f"User: {prompt}\nAssistant:"
    input_ids = tokenizer.encode(full_prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated_ids = []
    kv_caches = None
    prev_text = ""

    with torch.no_grad():
        out = model(input_tensor, use_cache=True)
        kv_caches = out["kv_caches"]
        next_logits = out["logits"][:, -1, :]

        for _ in range(max_tokens):
            logits = next_logits.clone()
            if generated_ids:
                recent = set(generated_ids[-64:])
                for tid in recent:
                    if tid < logits.size(-1):
                        logits[0, tid] = logits[0, tid] / rep_penalty if logits[0, tid] > 0 else logits[0, tid] * rep_penalty

            logits = logits / temperature
            logits = top_k_filter(logits, top_k)
            logits = top_p_filter(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            next_id_tensor = torch.multinomial(probs, num_samples=1)
            token_id = next_id_tensor[0, 0].item()

            if token_id == tokenizer.eos_id:
                break

            generated_ids.append(token_id)
            full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if "User:" in full_text:
                full_text = full_text[:full_text.index("User:")].strip()
                break

    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if "User:" in response:
        response = response[:response.index("User:")].strip()

    return jsonify({"response": response, "prompt": prompt})


@app.route("/api/stream", methods=["POST"])
def stream():
    data = request.json
    prompt = data.get("prompt", "").strip()
    temperature = float(data.get("temperature", 0.8))
    top_k = int(data.get("top_k", 40))
    top_p = float(data.get("top_p", 0.9))
    max_tokens = int(data.get("max_tokens", 200))
    rep_penalty = float(data.get("repetition_penalty", 1.3))

    if not prompt or model is None:
        return jsonify({"error": "Bad request"}), 400

    full_prompt = f"User: {prompt}\nAssistant:"

    def generate_stream():
        input_ids = tokenizer.encode(full_prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        generated_ids = []
        kv_caches = None
        prev_text = ""

        with torch.no_grad():
            out = model(input_tensor, use_cache=True)
            kv_caches = out["kv_caches"]
            next_logits = out["logits"][:, -1, :]

            for _ in range(max_tokens):
                logits = next_logits.clone()
                if generated_ids:
                    recent = set(generated_ids[-64:])
                    for tid in recent:
                        if tid < logits.size(-1):
                            logits[0, tid] = logits[0, tid] / rep_penalty if logits[0, tid] > 0 else logits[0, tid] * rep_penalty

                logits = logits / temperature
                logits = top_k_filter(logits, top_k)
                logits = top_p_filter(logits, top_p)
                probs = F.softmax(logits, dim=-1)
                next_id_tensor = torch.multinomial(probs, num_samples=1)
                token_id = next_id_tensor[0, 0].item()

                if token_id == tokenizer.eos_id:
                    break

                generated_ids.append(token_id)
                full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                new_part = full_text[len(prev_text):]
                prev_text = full_text

                if "User:" in full_text:
                    break

                if new_part:
                    yield f"data: {json.dumps({'token': new_part})}\n\n"

                next_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)
                out = model(next_tensor, use_cache=True, kv_caches=kv_caches)
                kv_caches = out["kv_caches"]
                next_logits = out["logits"][:, -1, :]

        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(generate_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/checkpoint_best.pt")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    load_model(args.checkpoint)

    print(f"\nChat UI ready at http://{args.host}:{args.port}", flush=True)
    print("Press Ctrl+C to stop\n", flush=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)