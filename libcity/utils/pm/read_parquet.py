import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import pandas as pd


class ReadParquet:
    def __init__(self, parquet_path):
        self.parquet_path = parquet_path

    def read_parquet(self, path):
        """读取 parquet 文件"""
        print(f"\n开始读取 {path} ...")
        try:
            df = pd.read_parquet(path, engine="fastparquet")  # ✅ 建议用 pyarrow,速度更快. 但是我这里使用fastparquet存储的，所以只有fastparquet才能保留list类型，pyarrow读取就是str
            # df = pd.read_parquet(path, engine="pyarrow")  # ✅ 建议用 pyarrow,速度更快. 但是我这里使用fastparquet存储的，所以只有fastparquet才能保留list类型，pyarrow读取就是str
            print(f"\n✅ parquet 读取:{path} 完成 , 行数: {len(df)}")
        except Exception as e:
            traceback.print_exc()
        return df

    def timer_task(self, stop_event):
        """计时任务，直到 stop_event 被设置"""
        start = time.time()
        while not stop_event.is_set():
            elapsed = time.time() - start
            print(f"\r处理中... {elapsed:.1f}s", end="", flush=True)
            time.sleep(1)
        print(f"\r处理结束，总耗时 {time.time() - start:.2f}s")

    def read_parquet_with_timer(self):
        """主函数：并行执行 计时 + 读取 parquet"""
        stop_event = threading.Event()

        with ThreadPoolExecutor(max_workers=2) as executor:
            read_future = executor.submit(self.read_parquet, self.parquet_path)
            timer_future = executor.submit(self.timer_task, stop_event)

            for future in as_completed([read_future]):
                df = future.result()
                stop_event.set()
                break

            timer_future.result()
            return df

    def save_parquet_with_timer(self, df):
        stop_event = threading.Event()

        def save_parquet(df):
            print(f"\n开始存储 ...")
            try:
                df.to_parquet(self.parquet_path, engine="fastparquet")
                print(f"\n✅ parquet 存储:{self.parquet_path} 完成 , 行数: {len(df)}")
            except Exception as e:
                traceback.print_exc()
                return False
            return True

        with ThreadPoolExecutor(max_workers=2) as executor:
            read_future = executor.submit(save_parquet, df)
            timer_future = executor.submit(self.timer_task, stop_event)

            for future in as_completed([read_future]):
                future.result()
                stop_event.set()
                break

            timer_future.result()
            return True

    def read_parquet_with_timer_assignpath(self, assigned_path):
        """主函数：并行执行 计时 + 读取 parquet"""
        stop_event = threading.Event()

        with ThreadPoolExecutor(max_workers=2) as executor:
            read_future = executor.submit(self.read_parquet, assigned_path)
            timer_future = executor.submit(self.timer_task, stop_event)

            for future in as_completed([read_future]):
                df = future.result()
                stop_event.set()
                break

            timer_future.result()
            return df

    def save_parquet_with_timer_assignpath(self, df, assigned_path):
        stop_event = threading.Event()

        def save_parquet(df):
            print(f"\n开始存储 ...")
            try:
                df.to_parquet(assigned_path, engine="fastparquet")
                print(f"\n✅ parquet 存储:{assigned_path} 完成 , 行数: {len(df)}")
            except Exception as e:
                traceback.print_exc()
                return False
            return True

        with ThreadPoolExecutor(max_workers=2) as executor:
            read_future = executor.submit(save_parquet, df)
            timer_future = executor.submit(self.timer_task, stop_event)

            for future in as_completed([read_future]):
                future.result()
                stop_event.set()
                break

            timer_future.result()
            return True

if __name__ == "__main__":
    parquet_path = r"D:\code\simple\raw_data\pm\chengdu\chengdu_trajs_2w_train.parquet"
    # h5_path = r"D:\code\dataset\porto_20200.h5"
    df = ReadParquet(parquet_path).read_parquet_with_timer()
    print(f"\nDataFrame shape: {df.shape}")
