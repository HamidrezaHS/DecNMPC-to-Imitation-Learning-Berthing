# مرحله چهارم پژوهش: IL چندعامله با انتخاب محلی مدل و پایش فرمان

## وضعیت و محدودیت مهم
این بسته یک **نسخه پژوهشی مستند همراه با پیش‌نویس مقاله Ocean Engineering** است. آزمون‌های نرم‌افزاری موفق‌اند، اما کنترلر هنوز در تمام شرایط به پهلوگیری موفق نمی‌رسد؛ هیچ گواهی ایمنی، آزمون مقیاس واقعی یا برتری آماری ادعا نمی‌شود. نسخه NMPC منتشرشده قبلی تغییری نکرده است.

## روش
- چهار عامل غیرمتمرکز؛ هر یدک‌کش حالت اندازه‌گیری‌شده کشتی، داده خودش، هدف و باد/جریان محلی را می‌خواند؛ وضعیت/فرمان دیگر یدک‌کش‌ها را نمی‌خواند.
- هر عامل بین Actor استاندارد (`il_results/stage3/actors_balanced`) و Actor محیط مخالف (`il_results/stage4/actors_paper_candidate`) انتخاب می‌کند؛ در مجموع هشت شبکه متعلق به چهار عامل.
- یک بازخورد محلی برای کمبود پیشرفت نزدیک اسکله و یک فیلتر **ابتکاری** کنترل سرعت/حاشیه هندسی به فرمان کمک می‌کند. به همین دلیل روش، BC خالص یا تضمین ریاضی ایمنی نیست.
- در **ارزیابی حلقه‌بسته** هیچ فرمانی از معلم NMPC دریافت نمی‌شود.

## نصب (PowerShell ویندوز)
```powershell
cd "<EXTRACTED_PROJECT_FOLDER>"
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-il.txt
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
.\.venv\Scripts\python.exe -m pytest -q tests tests_il
```
برای GPU، ابتدا نسخه CUDA سازگار با پایتورچ و سخت‌افزار را از pytorch.org نصب کن؛ آموزش روی CPU نیز پشتیبانی می‌شود.

## ارزیابی یک مانور و گرفتن شکل چهارزمانه
```powershell
python -m imitation.rollout `
  --actors il_results/stage4/actors_paper_candidate `
  --base-actors il_results/stage3/actors_balanced `
  --policy-type routed-neural `
  --progress-assist --progress-assist-toward-only `
  --safety-filter `
  --psi0 20 --wind-mps 8 --wind-to-deg 270 `
  --current-mps 0.25 --current-to-deg 270 `
  --duration 6500 --seed 123 `
  --out il_results/stage4/my_eval_270 --render
```
نتایج در پوشه خروجی شامل `metrics.json`، مسیر و فرمان‌ها و شکل‌های رسم‌شده خواهد بود. «موفق» یعنی ورود پایدار پنج‌ثانیه‌ای به محدوده پذیرش اولیه، نه نگه‌داری طولانی کنار فندر.

## بازتولید شش آزمون ازپیش‌تعریف‌شده
به‌دلیل هزینه محاسباتی، هر آزمون را در یک فرایند جدا اجرا کن؛ زمان سیستم عامل می‌تواند با لپ‌تاپ متفاوت باشد:
```powershell
python -m imitation.publication_suite `
  --protocol PAPER_PROTOCOL.json `
  --out il_results/stage4/my_prospective_reproduction `
  --methods baseline_stage3 stage4_routed `
  --group holdout --hard-timeout-s 360 --wall-limit-s 320
```
وقتی همه آزمایش‌ها تمام شوند، خروجی‌های `summary.csv`، `summary.json` و `score.json` را می‌بینی. **اگر watchdog فرایند را متوقف کند، آن مورد شکست/ناقص است و نباید موفق شمرده شود.** همچنین این آزمایش‌های قبلاً مشاهده‌شده را نمی‌توان دوباره به‌عنوان آزمون مستقل تازه معرفی کرد.

## تولید شکل‌های مقاله
```powershell
python -m imitation.publication_figures
python -m imitation.graphical_abstract
```
شکل‌های PDF/PNG در `manuscript_il/Figures/` و چکیده تصویری در `manuscript_il/Submission_Files/` ذخیره می‌شوند. فایل‌های خام لاگ هر نمودار در `il_results/stage4/` هستند.

## آموزش دوباره Actor تخصصی با داده‌های توسعه
```powershell
python -m imitation.train_paper `
  --dataset il_results/stage4/development_dagger_dataset.npz `
  --out il_results/stage4/my_retraining `
  --epochs 120 --device auto --seed 13
```
**توجه:** وزن‌های `actors_paper_candidate` از قبل در بسته‌اند. آموزش مجدد بدون تثبیت محیط محاسباتی می‌تواند وزن‌های متفاوت بسازد. تصمیم‌گیری درباره مدل نهایی باید بر مبنای ارزیابی حلقه‌بسته و مجموعه توسعه باشد، نه دست‌کاری مجموعه holdout گزارش‌شده.

## نتایج ثبت‌شده و منصفانه
| روش | موفق | TIMEOUT | UNSAFE |
|---|---:|---:|---:|
| IL متوازن قبلی با همان فیلتر محلی | ۲/۶ | ۳/۶ | ۱/۶ |
| IL انتخاب‌گر محیطی + اصلاح پیشرفت + همان فیلتر | ۴/۶ | ۲/۶ | ۰/۶ |

شش شرایط مختلف‌اند، نه شش بذر مستقل؛ هیچ ادعای معناداری آماری یا احتمال تضمینی ایمنی مجاز نیست. دو شکست روش جدید: شرایط A و D (آب آرام).

## پیش از ارسال مقاله
- داده‌ها و کد IL را در یک مخزن مستقل/نسخه جدید همراه با DOI منتشر کن. **DOI نسخه Dec-NMPC قبلی DOI این پژوهش IL نیست.**
- اطلاعات نویسنده مسئول، بودجه، تعارض منافع و سهم واقعی نویسندگان را تأیید کن.
- برای ادعای تعمیم گسترده باید آزمون‌های بیشتر، بذرهای مستقل، ابلیشن‌های کامل، آزمون پس از رسیدن به اسکله و شرایط حسگری/مدل سخت‌تر را اجرا کنی.

**Associated article:** From Decentralized NMPC to Low-Latency Imitation Learning for Cooperative Tugboat-Assisted Tanker Berthing
