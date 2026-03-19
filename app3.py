import os
import glob
import uvicorn
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
# 保持你的导入路径一致
from src.utils import SolarKnowledgeHandler

app = FastAPI()

# 获取路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 确保数据路径指向你存放 Excel 的实际位置
DATA_PATH = os.path.join(BASE_DIR, "data", "solar_stations", "*.xlsx")


@app.get("/search")
async def search_solar_data(query: str):
    try:
        files = glob.glob(DATA_PATH)
        if not files:
            return JSONResponse(status_code=404, content={"message": "未找到Excel文件"})

        all_results = []
        # 核心优化 1：仅处理前 1 个文件，且只读取前 1000 行
        for file in files[:1]:
            # nrows=1000 保证了读取速度，不会卡死
            df = pd.read_excel(file, nrows=1000).astype(str)

            # 核心优化 2：简单的关键词过滤
            mask = df.apply(lambda row: row.str.contains(query).any(), axis=1)
            matched_df = df[mask].head(5)  # 只取匹配的前 5 条

            if not matched_df.empty:
                records = matched_df.to_dict('records')
                all_results.extend(records)

        # 核心优化 3：如果没搜到，返回一个友好的提示而不是转圈
        if not all_results:
            return JSONResponse(content={"status": "empty", "message": "在前1000行中未找到结果，请尝试搜索 '2019'"})

        return JSONResponse(
            content={
                "status": "success",
                "results": all_results[:10]  # 严格限制返回条数
            }
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)