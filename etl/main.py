import argparse

from databricks.connect import DatabricksSession
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extracts image URLs from a source CSV, downloads images in parallel, "
            "and structures them in Volumes to be compatible with "
            "'torchvision.datasets.ImageFolder'."
        )
    )
    parser.add_argument("--catalog", default="marketplace", type=str)
    parser.add_argument("--schema", default="default", type=str)
    parser.add_argument(
        "--vfs_dir", default="/Volumes/marketplace/default/images", type=str
    )
    args = parser.parse_args()

    spark = DatabricksSession.builder.serverless().getOrCreate()
    spark.catalog.setCurrentCatalog(args.catalog)
    spark.catalog.setCurrentDatabase(args.schema)

    # Step 1: Ingest source data
    print("Step 1: Loading source images...")
    images_df: DataFrame = spark.read.csv(
        f"{args.vfs_dir}/df_images_src.csv.zst",
        header=True,
        inferSchema=True,
    ).select("category_name", "image_id", "image_url")
    images_df.write.mode("overwrite").saveAsTable("images_src")
    print(f"  Loaded {images_df.count()} images")

    # Step 2: Download images
    print("Step 2: Downloading images...")
    downloads_df: DataFrame = (
        spark.read.table("images_src")
        # .limit(100)
        .repartition(100)
        .mapInPandas(
            make_download_batch(args.vfs_dir),
            schema=(
                "category_name: string, image_id: string, "
                "image_url: string, status: string"
            ),
        )
    )
    downloads_df.write.mode("overwrite").saveAsTable("downloads")

    # Step 3: Generate stats
    print("Step 3: Generating statistics...")
    download_stats_df: DataFrame = downloads_df.groupBy(col("status")).agg(
        count("image_id").alias("count")
    )
    download_stats_df.write.mode("overwrite").saveAsTable("download_stats")
    download_stats_df.show()

    category_stats_df: DataFrame = (
        downloads_df.filter(col("status") == "success")
        .groupBy(col("category_name"))
        .agg(count("image_id").alias("count"))
    )
    category_stats_df.write.mode("overwrite").saveAsTable("category_stats")

    # export category stats to csv
    category_stats_df.coalesce(1).mapInPandas(
        make_export_csv(args.vfs_dir), schema=""
    ).collect()  # Force execution

    print("Done!")


def make_download_batch(vfs_dir: str):
    """
    Factory that creates download_batch UDF with imports inside
    (for cluster serialization).
    """

    def download_batch(iterator):
        import os

        import pandas as pd
        import requests

        for pdf in iterator:
            session = requests.Session()
            results: list[tuple[str, str, str, str]] = []
            for _, row in pdf.iterrows():
                category_name: str = row.category_name
                image_id: str = row.image_id
                image_url: str = row.image_url

                save_dir = f"{vfs_dir}/{category_name}"
                os.makedirs(save_dir, exist_ok=True)

                save_path = f"{save_dir}/{image_id}.jpg"

                status: str
                if os.path.exists(save_path):
                    status = "success"
                else:
                    try:
                        response = session.get(image_url, timeout=10)
                        if response.status_code == 200:
                            with open(save_path, "wb") as f:
                                f.write(response.content)
                            status = "success"
                        else:
                            status = f"error_{response.status_code}"
                    except Exception as e:
                        status = f"failed_{str(e)}"

                results.append((category_name, image_id, image_url, status))

            yield pd.DataFrame(
                results, columns=["category_name", "image_id", "image_url", "status"]
            )

    return download_batch


def make_export_csv(vfs_dir: str):
    """
    Factory that creates export_csv UDF with imports inside.
    """

    def export_csv(iterator):
        import pandas as pd

        df = pd.concat(iterator)
        df.to_csv(f"{vfs_dir}/df_categories.csv", index=False)
        yield pd.DataFrame()

    return export_csv


if __name__ == "__main__":
    main()
