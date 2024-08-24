import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="zhihan1996/DNA_bert_4")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=4, metadata={"help": "k-mer for input sequence. Default is 4."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    output_dir: str = field(default="output")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})

def generate_embeddings(sequence, tokenizer, model, device, kmer_size=4, max_length=512):
    # Generar k-mers de la secuencia
    tokens = [sequence[i:i+kmer_size] for i in range(len(sequence) - kmer_size + 1)]
    
    # Dividir tokens en fragmentos de longitud max_length
    num_chunks = (len(tokens) + max_length - 1) // max_length
    embeddings_list = []
    
    for i in range(num_chunks):
        chunk_tokens = tokens[i * max_length:(i + 1) * max_length]
        inputs = tokenizer(chunk_tokens, return_tensors='pt', padding=True,  max_length=max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Mover inputs al dispositivo (GPU/CPU)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Calcular el promedio del último estado oculto y mover al CPU
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings_list.append(embeddings)
    
    return np.mean(embeddings_list, axis=0)  # Promediar embeddings de todos los fragmentos
    
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(f"El k-mer proporcionado es: {data_args.kmer}, la ruta de los datos es {data_args.data_path}")
    print("Cargando tokenizer y modelo...")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Leyendo archivo de datos...")
    df = pd.read_parquet(data_args.data_path)

    print("Generando k-mers...")
    df['tokens'] = df['sequence'].apply(lambda x: [x[i:i+data_args.kmer] for i in range(len(x) - data_args.kmer + 1)])

    print("Generando embeddings...")
    embeddings_list = []
    for tokens in tqdm(df['tokens'], desc="Generando embeddings"):
        embeddings = generate_embeddings(' '.join(tokens), tokenizer, model, device, kmer_size=data_args.kmer, special_token=special_token)
        embeddings_list.append(embeddings)

    df['embeddings'] = embeddings_list

    print("Guardando tokens y embeddings en archivo Parquet...")
    output_path = os.path.join(training_args.output_dir, "embeddings_with_tokens.parquet")
    df.to_parquet(output_path)

    print(f"Tokens y embeddings guardados en {output_path}")

if __name__ == "__main__":
    main()
