from multiprocessing import Process, Pool
from ..generate.constant import OutputType


def _get_ailake_table_length(table_name):
    from lake.table import ScanTableReader, closePanguFs
    import mdl_utils
    lake_config = mdl_utils.lake_path_to_lake_config(table_name)
    reader = ScanTableReader(lake_config, 2000)
    reader.open_slice(0, 1)
    row_count = reader.count_records()
    reader.close()
    closePanguFs()
    return row_count

def _get_ailake_schema(table_name):
    from lake.table import ScanTableReader, closePanguFs
    import mdl_utils
    lake_config = mdl_utils.lake_path_to_lake_config(table_name)
    reader = ScanTableReader(lake_config, 2000)
    reader.open_slice(0, 1)
    columns = reader.get_schema()
    reader.close()
    closePanguFs()
    return columns

def _exist_table(table_name):
    from lake.table import ScanTableReader, closePanguFs
    from lake.exception import LakeNotexistException
    import mdl_utils
    lake_config = mdl_utils.lake_path_to_lake_config(table_name)
    reader = ScanTableReader(lake_config, 2000)
    try:
        reader.open_slice(0, 1)
        reader.close()
        closePanguFs()
        return True
    except LakeNotexistException:
        closePanguFs()
        return False

def get_ailake_table_length(table_name):
    # ailake执行多进程读取要求在dataloader fork之前主进程不能有盘古操作，所以获取lake meta信息放在子进程中获取
    with Pool(processes=1) as pool:
        result = pool.apply_async(_get_ailake_table_length, (table_name,))
        row_count = result.get()
        return row_count

def get_ailake_columns_name(table_name):
        # ailake执行多进程读取要求在dataloader fork之前主进程不能有盘古操作，所以获取lake meta信息放在子进程中获取
    with Pool(processes=1) as pool:
        result = pool.apply_async(_get_ailake_schema, (table_name,))
        columns = result.get()
        return columns[0]


def get_ailake_schema(table_name):
        # ailake执行多进程读取要求在dataloader fork之前主进程不能有盘古操作，所以获取lake meta信息放在子进程中获取
    with Pool(processes=1) as pool:
        result = pool.apply_async(_get_ailake_schema, (table_name,))
        columns = result.get()
        return columns


def exist_table(table_name):
        # ailake执行多进程读取要求在dataloader fork之前主进程不能有盘古操作，所以获取lake meta信息放在子进程中获取
    with Pool(processes=1) as pool:
        return pool.apply_async(_exist_table, (table_name,)).get()


def gen_output_table_schema_in_ailake(table_name, select_col_indexes=[], generate_num=1,
                            output_type=OutputType.Sequence):
    from odps.models import Column, Partition
    from odps.models.table import TableSchema
    partition = Partition(name='ds', type='string', comment='the partition')


    table_schema = get_ailake_schema(table_name)
    columns = []
    for i, table_column_name in enumerate(table_schema[0]):
        if select_col_indexes and i not in select_col_indexes:
            continue
        column_item = Column(name=table_column_name, type=table_schema[1][i])
        columns.append(column_item)
    if output_type in (OutputType.Sequence, OutputType.TokensProbabilities):
        # TODO: output_scores的新加列需要改成array<double>
        columns.append(Column(name="generate_results", type="string" if generate_num == 1 else "array<string>"))
    elif output_type == OutputType.Embedding:
        columns.append(Column(name="generate_embedding", type="array<double>"))
    elif output_type == OutputType.RewardScores:
        columns.append(Column(name="generate_reward_scores", type="double"))
    else:
        raise Exception(f"Unknown output_type:{output_type}")

    if output_type == OutputType.TokensProbabilities:
        columns.append(Column(name="output_scores", type="string"))
    output_table_schema = TableSchema(columns=columns, partitions=[partition])
    return output_table_schema


def split_ailake_bytes_data(data):
    if b'---mdl_ailake_image_split_flags---' in data:
        return data.split(b'---mdl_ailake_image_split_flags---')
    elif b'---nebula_sql_ailake_image_split_flags---' in data:
        return data.split(b'---nebula_sql_ailake_image_split_flags---')
    else:
        return [data]

