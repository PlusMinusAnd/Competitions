import subprocess
import os
import shutil

# ===== 설정 =====
script = "./Drug/_code/Set2set_run_until_good.py"               # 실행할 스크립트 파일명
score_threshold = 0.62463           # 원하는 최소 점수
r = 1                               # 시작 시드

while True:
    print(f"\n🔁 Try with random_state={r}...")

    # 스크립트 실행
    result = subprocess.run(
        ["python", script, str(r)],
        capture_output=True, text=True
    )

    # 로그 저장
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/log_r{r}.txt", "w", encoding="utf-8") as f:
        f.write(result.stdout + "\n==== stderr ====\n" + result.stderr)

    # 출력에서 Score📈 줄 찾기
    lines = result.stdout.splitlines()
    score_lines = [line for line in lines if "Score📈" in line]

    if score_lines:
        last = score_lines[-1]
        try:
            score_val = float(last.split("Score📈")[-1].strip().replace(":", ""))
            print(f"✅ Score = {score_val:.5f}")

            if score_val >= score_threshold:
                print(f"🎯 목표 점수 달성! → {score_val:.5f} ≥ {score_threshold}")
                break
            else:
                print(f"⚠️ 점수 미달: {score_val:.5f} < {score_threshold}")
                shutil.rmtree("./Drug/_02/full_pipeline", ignore_errors=True)
        except Exception as e:
            print(f"❌ 점수 파싱 실패: {e}")
    else:
        print("❌ Score📈 출력 없음. stderr 확인 요망")

    r += 1
