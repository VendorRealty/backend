# RenderRealty backend png to stl utility
## setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## run
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## convert to stl
```bash
curl -fS -X POST \                                                                                        -F "image=@./images/test-plan.png;type=image/png" \                                                                                        -F "max_height_mm=10" -F "scale_mm_per_px=0.2" -F "downsample=1024" \                                                                                        http://localhost:8000/api/convert -o test_plan.stl
```

## to see the mask
```bash
curl -fS -X POST \                                                                                        -F "image=@./images/test-plan.png;type=image/png" \                                                                                        -F "max_height_mm=10" -F "scale_mm_per_px=0.2" -F "downsample=1024" \                                                                                        -F "debug=mask" \                                                                                        http://localhost:8000/api/convert -o test_plan.stl
```
