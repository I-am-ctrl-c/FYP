# json_pager.py
# 流式分页 + 结构探针（不加载整文件）
# 依赖: pip install ijson

import sys, json, argparse, itertools
from typing import Iterable, Tuple, Any

def die(msg: str, code: int = 1):
    print(f"[ERROR] {msg}", file=sys.stderr); sys.exit(code)

def detect_root(path: str) -> str:
    """返回 'object' | 'array'"""
    import ijson
    with open(path, 'rb') as f:
        for prefix, ev, val in ijson.parse(f):
            if ev == 'start_map':   return 'object'
            if ev == 'start_array': return 'array'
    return 'unknown'

def truncate(x: Any, maxlen: int = 200) -> str:
    s = json.dumps(x, ensure_ascii=False)
    return s if len(s) <= maxlen else s[:maxlen] + "…"

def safe_get(d: dict, dotted: str):
    """从 dict 里按 a.b.c 路径取值"""
    cur = d
    for p in dotted.split('.'):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur

def iter_object_records(path: str) -> Iterable[Tuple[str, dict]]:
    """顶层是对象：yield (key, record_dict)"""
    import ijson
    with open(path, 'rb') as f:
        for k, v in ijson.kvitems(f, ''):
            yield k, v

def iter_array_records(path: str) -> Iterable[dict]:
    """顶层是数组：yield record_dict"""
    import ijson
    with open(path, 'rb') as f:
        for v in ijson.items(f, 'item'):
            yield v

def print_schema_events(path: str, max_events: int = 200, max_depth: int = 3):
    """事件流探针：打印前 N 个事件（限制深度），用于“看结构”而不构建大对象。"""
    import ijson
    depth = 0
    printed = 0
    with open(path, 'rb') as f:
        for prefix, ev, val in ijson.parse(f):
            # prefix 用 '.' 分层
            depth = 0 if prefix == '' else prefix.count('.')
            if depth <= max_depth:
                if ev in ('map_key',):
                    line = f"{prefix} {ev} {val}"
                elif ev in ('string','number','boolean','null'):
                    line = f"{prefix} {ev} {truncate(val, 120)}"
                else:
                    line = f"{prefix} {ev}"
                print(line)
                printed += 1
                if printed >= max_events:
                    break

def page_print_object(path: str, page: int, size: int, fields: list):
    """顶层 object 的分页打印"""
    start = (page - 1) * size
    end   = start + size
    rows = []
    for idx, (k, rec) in enumerate(iter_object_records(path)):
        if idx < start:
            continue
        if idx >= end:
            break
        if fields:
            row = {"__key__": k}
            for f in fields:
                row[f] = safe_get(rec, f)
        else:
            # 默认打印简要字段（不建议全打印）
            row = {"__key__": k}
            for f in ('frames','fps','roi_size','roi_wh','method'):
                if isinstance(rec, dict) and f in rec:
                    row[f] = rec[f]
        rows.append(row)
    if not rows:
        print(f"[INFO] 没有更多记录（page={page}, size={size}）")
    else:
        for r in rows:
            print(truncate(r, 500))

def page_print_array(path: str, page: int, size: int, fields: list):
    """顶层 array 的分页打印"""
    start = (page - 1) * size
    end   = start + size
    rows = []
    for idx, rec in enumerate(iter_array_records(path)):
        if idx < start:
            continue
        if idx >= end:
            break
        if fields:
            row = {}
            for f in fields:
                row[f] = safe_get(rec, f)
        else:
            row = {}
            for f in ('frames','fps','roi_size','roi_wh','method'):
                if isinstance(rec, dict) and f in rec:
                    row[f] = rec[f]
        row['__idx__'] = idx
        rows.append(row)
    if not rows:
        print(f"[INFO] 没有更多记录（page={page}, size={size}）")
    else:
        for r in rows:
            print(truncate(r, 500))

def list_top_keys(path: str, limit: int):
    """仅列出顶层 key（适用于顶层 object），不读取值体"""
    import ijson
    with open(path, 'rb') as f:
        n = 0
        for prefix, ev, val in ijson.parse(f):
            if prefix == '' and ev == 'map_key':
                print(val)
                n += 1
                if n >= limit:
                    break

def show_first_record_fields(path: str, root: str, limit_fields: int = 50):
    """打印第一条记录的字段名（不深入展开大型数组）"""
    if root == 'object':
        gen = iter_object_records(path)
        try:
            k, v = next(gen)
        except StopIteration:
            die("空对象")
        print(f"[ROOT=object] TOP_KEY = {k}")
        if isinstance(v, dict):
            keys = list(v.keys())
            print("FIELDS:", keys[:limit_fields])
    elif root == 'array':
        gen = iter_array_records(path)
        try:
            v = next(gen)
        except StopIteration:
            die("空数组")
        if isinstance(v, dict):
            keys = list(v.keys())
            print("[ROOT=array]")
            print("FIELDS:", keys[:limit_fields])
    else:
        die("无法识别根类型")

def main():
    ap = argparse.ArgumentParser(
        description="超大 JSON 流式分页与结构查看（不加载整文件）"
    )
    ap.add_argument("-f", "--file", required=True, help="JSON 文件路径")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 结构探针（事件流）
    sp_schema = sub.add_parser("schema", help="事件流探针：打印前若干事件（用于看结构）")
    sp_schema.add_argument("--events", type=int, default=200, help="打印事件条数（默认200）")
    sp_schema.add_argument("--depth", type=int, default=3, help="最大层级深度（默认3）")

    # 顶层 key 列表（仅 object）
    sp_keys = sub.add_parser("keys", help="列出顶层 key（顶层是对象时有效）")
    sp_keys.add_argument("--limit", type=int, default=100, help="最多列出多少个 key（默认100）")

    # 第一条记录字段
    sp_first = sub.add_parser("first", help="显示第一条记录的字段名（object: 第一对 K/V；array: 第一项）")
    sp_first.add_argument("--limit-fields", type=int, default=50, help="最多显示多少字段名（默认50）")

    # 分页
    sp_page = sub.add_parser("page", help="分页打印（默认只挑常见字段；可用 --fields 指定）")
    sp_page.add_argument("--page", type=int, default=1, help="第几页（从1开始）")
    sp_page.add_argument("--size", type=int, default=10, help="每页多少条")
    sp_page.add_argument("--fields", type=str, default="", help="逗号分隔的字段路径（a.b.c），空则打印常见字段")

    args = ap.parse_args()
    path = args.file

    try:
        import ijson  # noqa: F401
    except Exception:
        die("缺少依赖 ijson，请先运行：pip install ijson")

    root = detect_root(path)
    if root not in ("object", "array"):
        die("无法识别根类型（既不是对象也不是数组）")

    if args.cmd == "schema":
        print(f"[INFO] root={root}")
        print_schema_events(path, max_events=args.events, max_depth=args.depth)

    elif args.cmd == "keys":
        if root != "object":
            die("顶层不是对象：keys 仅适用于顶层 object")
        list_top_keys(path, args.limit)

    elif args.cmd == "first":
        show_first_record_fields(path, root, args.limit_fields)

    elif args.cmd == "page":
        fields = [s.strip() for s in args.fields.split(",") if s.strip()]
        print(f"[INFO] root={root} page={args.page} size={args.size} fields={fields or '[frames,fps,roi_size,roi_wh,method]'}")
        if root == "object":
            page_print_object(path, args.page, args.size, fields)
        else:
            page_print_array(path, args.page, args.size, fields)

if __name__ == "__main__":
    main()
