import logging
import os
import time

import pandas as pd

from agent import (
    AgentState,  # Agent graph'ını import ediyoruz
    app,
)

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_benchmark(csv_path="assets/projects.csv", sample_size=3):
    """
    CSV dosyasından rastgele projeler seçip agent'ı test eder.
    """
    if not os.path.exists(csv_path):
        logger.error(f"CSV dosyası bulunamadı: {csv_path}")
        return

    logger.info(f"Benchmark başlatılıyor... {csv_path} okunuyor.")

    try:
        df = pd.read_csv(csv_path)
        # Sadece Github URL'si olanları al
        df = df[df["Github URL"].notna()]

        # Rastgele örneklem al (Demo süresi kısıtlı olduğu için az sayıda seçiyoruz)
        sample = df.sample(n=min(sample_size, len(df)), random_state=42)

        results = []

        logger.info(f"Toplam {len(sample)} proje üzerinde test yapılacak.")

        for index, row in sample.iterrows():
            project_title = row.get("Title", "Unknown Project")
            github_url: str = str(row.get("Github URL"))

            logger.info(f"Testing: {project_title} ({github_url})")

            start_time = time.time()
            status = "Failed"

            try:
                # AgentState'i hazırla
                initial_state: AgentState = {
                    "article_url": github_url,  # CSV'deki github linkini kullanıyoruz
                    "github_links": [],
                    "file_paths": [],
                    "ai_response": "",
                }

                # Agent'ı çalıştır
                output = app.invoke(initial_state)

                # Başarılı olup olmadığını kontrol et (Dosya üretilmiş mi?)
                if output.get("create_files"):
                    status = "Success"
                else:
                    status = "No Files Created"

            except Exception as e:
                logger.error(f"Hata oluştu ({project_title}): {e}")
                status = f"Error: {str(e)}"

            duration = time.time() - start_time
            results.append(
                {
                    "Project": project_title,
                    "URL": github_url,
                    "Status": status,
                    "Duration": f"{duration:.2f}s",
                }
            )

            logger.info(f"Result for {project_title}: {status}")
            logger.info("-" * 30)

        # Sonuç raporu
        logger.info("\n=== BENCHMARK REPORT ===")
        results_df = pd.DataFrame(results)
        print(results_df)

        # Sonuçları kaydet
        results_df.to_csv("benchmark_results.csv", index=False)
        logger.info("Sonuçlar benchmark_results.csv dosyasına kaydedildi.")

    except Exception:
        logger.exception("Benchmark sırasında kritik hata:")


if __name__ == "__main__":
    # Test etmek için 3 proje seçiyoruz. Süreye göre artırabilirsin.
    run_benchmark(sample_size=3)
