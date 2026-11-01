# نسخه پژوهشی قابل اجرا برای Applied Ocean Research و پایه Meta-RL

این بسته حاوی **همان سیاست‌های IL موجود**، چهار Actor مستقل، مدل کوپل‌شده
پنج شناور و معلم Dec-NMPC است. کدهای اصلی کنترلر و مدل فیزیکی تغییر نکرده‌اند.
ابزارهای اضافه‌شده برای **ارزیابی علمی گسترده** و آماده‌سازی آینده Meta-RL
هستند. نتایج ۱۰ سناریوی جدید هنوز اجرا نشده‌اند و نباید در مقاله نوشته شوند.

## مراحل نصب روی Windows 11 / PowerShell

1. فایل ZIP را استخراج کن. پوشه‌ای را باز کن که `requirements-il.txt` در آن است.
2. در نوار مسیر Explorer بنویس `powershell` و Enter بزن.
3. پایتون نسخه 3.11 تا 3.13 و Git را داشته باش؛ تطابق نسخه PyTorch و CUDA را از سایت رسمی PyTorch بررسی کن.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-il.txt
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
.\.venv\Scripts\python.exe -m research_protocol.preflight
.\.venv\Scripts\python.exe -m pytest -q tests tests_il
```

GPU لپ‌تاپ RTX 3050 Ti با حافظه ۴ گیگابایت برای آموزش شبکه‌های کوچک قابل
استفاده است، اگر درایور و PyTorch CUDA سازگار باشند. تأیید GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

برای شبیه‌سازی و مقایسه Dec-NMPC استفاده از CPU و محدودکردن نخ‌ها بهتر
است؛ برای Actor training در صورت دسترس بودن CUDA، `--device auto` قابل
استفاده است. بنچمارک‌های زمان محاسبات به سخت‌افزار وابسته هستند و باید
روی لپ‌تاپ **دوباره اندازه‌گیری** شوند.

## اجرای سریع برای اطمینان از درست بودن نرم‌افزار

```powershell
.\.venv\Scripts\python.exe -m research_protocol.runner --dry-run --limit-tasks 2 --seeds 1049
.\.venv\Scripts\python.exe -m research_protocol.runner --methods routed_il baseline_il teacher_nmpc --limit-tasks 1 --seeds 1049 --duration 20 --hard-timeout-s 180 --out results/check_only
.\.venv\Scripts\python.exe -m research_protocol.report --results results/check_only
```

این **صرفاً تست فنی ۲۰ ثانیه‌ای** است، نتیجه مقاله محسوب نمی‌شود.

## اجرای مطالعه اصلی چندبذری (بعد از ثابت‌کردن همه پارامترها)

```powershell
.\.venv\Scripts\python.exe -m research_protocol.runner --methods routed_il baseline_il teacher_nmpc --out results/prospective_full --hard-timeout-s 3600
.\.venv\Scripts\python.exe -m research_protocol.report --results results/prospective_full
.\.venv\Scripts\python.exe -m research_protocol.latex_summary --results results/prospective_full --out results/prospective_full/table_for_manuscript.tex
```

مجموعه پیشنهادی شامل ۱۰ سناریو با ۳ seed و سه روش است؛ اجراها می‌توانند
محاسبات سنگینی باشند. می‌توان با `--limit-tasks 2 --seeds 1049` از نمونه
کوچک‌تر شروع کرد و سپس پروتکل کامل را اجرا کرد. برای کنترل بهتر، از پوشه
خروجی متفاوت برای آزمون‌های توسعه‌ای و نهایی استفاده کن. اجرای اصلی را
بعد از دیدن نتایج برای تنظیم آستانه‌ها **تغییر نده**؛ در غیر این صورت آن
آزمون‌ها دیگر مستقل نیستند.

فایل‌های `records.csv`, `records.json`, `audit.json`, `protocol_frozen.json`,
`analysis_summary.json`, `all_outcomes.pdf` و هر `metrics.json` داخل پوشه
خروجی ساخته می‌شوند. در هر اجرا `SUCCESS`, `TIMEOUT`, `UNSAFE`,
`PROCESS_TIMEOUT` و `PROCESS_ERROR` جدا ثبت می‌شوند. توقف به‌عنوان موفقیت
شمرده نمی‌شود.

**قید مقایسه منصفانه:** تغییر جرم ۱۰٪ در یکی از سناریوها فقط برای IL قابل
آزمون است؛ رابط معلم Dec-NMPC همان تغییر فیزیکی را از خط فرمان نمی‌پذیرد.
لذا آن جفت با `UNSUPPORTED_COMPARISON` نشانه‌گذاری می‌شود و نباید در
مقایسه همسان معلم/دانش‌آموز استفاده شود.

## تجدید بنچمارک سرعت روی لپ‌تاپ

```powershell
.\.venv\Scripts\python.exe benchmark/benchmark_decision_latency.py --root . --out benchmark/my_laptop --cycles 18
```

عدد 48.4 برابر که در مقاله آمده، اندازه‌گیری دستگاه قبلی است. نتیجه
لپ‌تاپ ممکن است فرق کند؛ خروجی جدید را جدا نگه دار و فقط پس از کنترل
روش اندازه‌گیری در مقاله استفاده کن.

## آموزش مجدد Actorها (اختیاری؛ به‌جای وزن‌های ثابت مقاله استفاده نشود)

```powershell
.\.venv\Scripts\python.exe -m imitation.train_paper --dataset il_results/stage4/development_dataset.npz --out results/my_retraining --epochs 80 --device auto
```

اگر وزن‌های جدید را آموزش بدهی، باید آزمون‌های مجدد روی مجموعه **تازه**
اجرا شود. نتایج مقاله پیشین به وزن‌های جدید قابل نسبت‌دادن نیست.

## پایه مطالعات Meta-RL در آینده (صرفاً ابزار آماده‌سازی)

```powershell
.\.venv\Scripts\python.exe -m meta_rl_bridge.tasks --out experiments/future_meta_tasks.json
.\.venv\Scripts\python.exe -m meta_rl_bridge.episode --run il_results/stage4/frozen_holdout_stage4/stage4_routed/B_105_75_h29 --out results/meta_history_example
```

این خروجی صرفاً چهار تاریخچه مشاهده محلی است، نه Meta-RL یا سازگاری
آموخته‌شده. پارامترهای ناشناخته دینامیک کشتی و اطلاعات سایر یدک‌کش‌ها
نباید به Actor در زمان اجرا داده شوند.

## پیش از ارسال مقاله

- DOI قدیمی `10.5281/zenodo.23831645` مربوط به معلم Dec-NMPC است.
- IL باید GitHub Repository و Zenodo Release **مستقل** و شناسه DOI واقعی داشته باشد.
- مسئول مقاله باید جداول جدید را فقط پس از اجرای کامل و تأیید لاگ‌ها وارد کند.
- بیشترین اولویت علمی، مقایسه چندبذری با معلم در همان سناریوها و گزارش
  مشکلات آب آرام و شرایط محیطی شدید است؛ هیچ تضمین ایمنی ادعا نشده است.

**Associated article:** From Decentralized NMPC to Low-Latency Imitation Learning for Cooperative Tugboat-Assisted Tanker Berthing
