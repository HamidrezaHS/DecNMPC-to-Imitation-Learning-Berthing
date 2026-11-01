# راهنمای انتشار GitHub/Zenodo برای مقاله IL

**عنوان مقاله:** From Decentralized NMPC to Low-Latency Imitation Learning for Cooperative Tugboat-Assisted Tanker Berthing

این بسته **فقط** برای انتشار کد، داده‌های اجرای عددی، وزن‌های Actor و اسکریپت‌های بازتولید است؛ فایل‌های ارسالی مجله در ZIP جدا قرار دارند.

## قبل از Public کردن

1. مالکیت و مجوز انتشار کدهای مرجع و فایل‌های هندسه را با سایر صاحبان حقوق بررسی کنید.
2. فایل `LICENSE_DECISION_REQUIRED.md` را بخوانید؛ مجوز کد IL از پیش به‌صورت خودکار انتخاب نشده است.
3. از نبودن اطلاعات محرمانه و فایل‌های شخصی مطمئن شوید.

## مراحل GitHub و Zenodo

1. یک مخزن مجزای IL در GitHub بسازید و محتوای استخراج‌شده ZIP را در ریشه مخزن وارد کنید.
2. فایل `README.md`، `.gitignore`، `CITATION.cff` و دایرکتوری‌های کد، داده و وزن‌ها را حفظ کنید.
3. آزمون‌ها: `python -m pytest -q tests tests_il`.
4. ارزیابی تقلید: `python -m reproducibility.evaluate_teacher_imitation --out benchmark/imitation_quality`.
5. ارزیابی سناریوها و مقایسه Dec-NMPC را طبق `README.md` اجرا کنید؛ شرایط آزمون مقاله را تغییر ندهید.
6. مخزن Public را پیش از Release به Zenodo متصل کنید.
7. Release با Tag `v1.0.0` منتشر کنید و DOI مخصوص IL را از Zenodo دریافت کنید.
8. همان DOI واقعی را در `CITATION.cff` و در بخش Data and software availability مقاله درج کنید.

**توجه:** DOI قبلی `10.5281/zenodo.23831645` فقط برای کد مرجع Dec-NMPC است. اینجا DOI ساختگی ننوشته‌ایم.

## اسکریپت ویدیو

برای بازتولید فایل تصویری مقاله، `ffmpeg` در PATH باشد:

`python -m reproducibility.make_supplementary_video --fps 24 --out supplementary/Video_S1_Berthing_Comparison.mp4`

بازپخش فقط از لاگ‌های عددی ساخته شده و درون‌یابی نمایشی دارد.

## Release title (copy exactly)

From Decentralized NMPC to Low-Latency Imitation Learning for Cooperative Tugboat-Assisted Tanker Berthing — research software v1.0.0
