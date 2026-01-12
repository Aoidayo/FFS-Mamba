import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import pandas as pd


class ReadH5:
    def __init__(self, h5_path):
        """
        Args:
            h5_path (str): HDF5 文件路径
        """
        self.h5_path = h5_path

    def list_keys(self, path):
        """列出 HDF5 文件中所有的 key"""
        print(f"\n开始扫描 {path} 中的 keys ...")
        try:
            with pd.HDFStore(path, mode="r") as store:
                keys = store.keys()
            print(f"\n✅ HDF5 文件中共 {len(keys)} 个 key:")
            for k in keys:
                print(f"  - {k}")
            return keys
        except Exception:
            traceback.print_exc()
            return []

    def timer_task(self, stop_event):
        """计时任务"""
        start = time.time()
        while not stop_event.is_set():
            elapsed = time.time() - start
            print(f"\r处理中... {elapsed:.1f}s", end="", flush=True)
            time.sleep(1)
        print(f"\r处理结束，总耗时 {time.time() - start:.2f}s")

    def list_keys_with_timer(self):
        """并行执行 计时 + 列出所有 keys"""
        stop_event = threading.Event()

        with ThreadPoolExecutor(max_workers=2) as executor:
            list_future = executor.submit(self.list_keys, self.h5_path)
            timer_future = executor.submit(self.timer_task, stop_event)

            for future in as_completed([list_future]):
                keys = future.result()
                stop_event.set()
                break

            timer_future.result()
            return keys

    def read_h5(self, path, key):
        """读取 HDF5 文件中指定 key"""
        print(f"\n开始读取 {path} (key={key}) ...")
        try:
            df = pd.read_hdf(path, key=key)
            print(f"\n✅ HDF5 读取:{path} 完成 , 行数: {len(df)}")
        except Exception:
            traceback.print_exc()
            df = pd.DataFrame()
        return df

    def read_h5_with_timer(self, key):
        """带计时读取"""
        stop_event = threading.Event()

        with ThreadPoolExecutor(max_workers=2) as executor:
            read_future = executor.submit(self.read_h5, self.h5_path, key)
            timer_future = executor.submit(self.timer_task, stop_event)

            for future in as_completed([read_future]):
                df = future.result()
                stop_event.set()
                break

            timer_future.result()
            return df

    def save_h5_with_timer(self, df, key, mode="w"):
        """保存 HDF5 文件"""
        stop_event = threading.Event()

        def save_h5(df):
            print(f"\n开始存储 ...")
            try:
                df.to_hdf(self.h5_path, key=key, mode=mode, format="table")
                print(f"\n✅ HDF5 存储:{self.h5_path} 完成 , 行数: {len(df)}")
            except Exception:
                traceback.print_exc()
                return False
            return True

        with ThreadPoolExecutor(max_workers=2) as executor:
            write_future = executor.submit(save_h5, df)
            timer_future = executor.submit(self.timer_task, stop_event)

            for future in as_completed([write_future]):
                future.result()
                stop_event.set()
                break

            timer_future.result()
            return True


if __name__ == "__main__":
    h5_path = r"D:\code\dataset\porto_20200.h5"
    reader = ReadH5(h5_path)

    # ✅ 列出所有 keys 并计时
    keys = reader.list_keys_with_timer()

    if not keys:
        print("\n⚠️ 没有找到任何 key。")
    else:
        # ✅ 示例：读取第一个 key
        key = keys[0]
        df = reader.read_h5_with_timer(key)
        print(f"\nDataFrame shape: {df.shape}")
