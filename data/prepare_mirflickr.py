import argparse
from dataclasses import replace

from classic_hashing.config import ConfigNode
from classic_hashing.data.prepare_dataset import build_split, convert_dataset
from classic_hashing.data.registry import get_dataset_spec


def mgsh_split(sample_count, query_size=1900, train_size=9500):
    spec = replace(
        get_dataset_spec("mirflickr"),
        query_size=query_size,
        train_size=train_size,
        database_start=query_size,
    )
    return build_split(spec, sample_count)


def convert_mirflickr(
    image_mat,
    text_mat,
    label_mat,
    output_path,
    split_path,
    query_size=1900,
    train_size=9500,
    image_dim=4096,
    text_dim=1386,
    num_classes=24,
):
    config = ConfigNode(
        {
            "dataset": ConfigNode(
                {
                    "name": "mirflickr",
                    "image_path": str(image_mat),
                    "text_path": str(text_mat),
                    "label_path": str(label_mat),
                    "h5_path": str(output_path),
                    "split_path": str(split_path),
                    "image_dim": image_dim,
                    "text_dim": text_dim,
                    "num_classes": num_classes,
                }
            )
        }
    )
    return convert_dataset(
        config,
        protocol_overrides={
            "query_size": query_size,
            "train_size": train_size,
            "database_start": query_size,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-mat", required=True)
    parser.add_argument("--text-mat", required=True)
    parser.add_argument("--label-mat", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-output", required=True)
    parser.add_argument("--query-size", type=int, default=1900)
    parser.add_argument("--train-size", type=int, default=9500)
    args = parser.parse_args()
    print(
        convert_mirflickr(
            args.image_mat,
            args.text_mat,
            args.label_mat,
            args.output,
            args.split_output,
            args.query_size,
            args.train_size,
        )
    )


if __name__ == "__main__":
    main()
