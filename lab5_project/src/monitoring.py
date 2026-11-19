"""ЛР5: моніторинг моделі. Тягне reference/current з БД, викликає Evidently, зберігає HTML-дашборди в reports/.

Запуск (з кореня проєкту):
    python src/monitoring.py                    # усі 4 звіти, current = усі inference-запити
    python src/monitoring.py --run drift        # лише прогін зі змодельованим дрейфом
    python src/monitoring.py --run normal --reports drift performance
Автоматизація (бонус): запускайте за розкладом (cron / Планувальник завдань Windows) або викликайте GET /monitor.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services import monitoring as m  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="all", help="all | normal | drift")
    ap.add_argument("--reports", nargs="+", default=list(m.REPORT_TYPES), choices=m.REPORT_TYPES)
    args = ap.parse_args()

    ref, cur = m.load_reference(), m.load_current(args.run)
    print(f"reference: {ref.shape}, current ({args.run}): {cur.shape}, current з ground truth: {cur['target'].notna().sum()}")
    print("\nШвидка перевірка дрейфу (scipy):\n", m.drift_summary(ref, cur).round(4).to_string(index=False))
    print("\nЧастки класів:\n", m.class_shares(ref, cur).to_string())
    print("\nЯкість моделі:\n", m.performance_summary(ref, cur).to_string(index=False))

    for kind in args.reports:
        try:
            path = m.build_report(kind, ref, cur, m.REPORTS_DIR / f"{kind}_{args.run}.html")
            print(f"[OK] {kind}: {path}")
        except Exception as e:      # один збійний звіт не повинен зупиняти решту
            print(f"[ПОМИЛКА] {kind}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
