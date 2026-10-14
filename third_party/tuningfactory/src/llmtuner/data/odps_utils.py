import os

from ..generate.constant import OutputType


def parse_table_path(odps_table_path):
    """Method that parse odps table path.
    """
    str_list = odps_table_path.split("/")
    if len(str_list) < 5 or str_list[3] != "tables":
        raise ValueError(
            "'%s' is invalid, please refer: 'odps://${your_projectname}/"
            "tables/${table_name}/${pt_1}/${pt_2}/...'" % odps_table_path)

    table_partition = ",".join(str_list[5:])
    if not table_partition:
        table_partition = None
    return (str_list[2], str_list[4], table_partition)


def gen_output_table_schema(
    table_schema, select_col_indexes=[], generate_num=1, output_type=OutputType.Sequence, field_mapping=None
):
    from odps.models import Column
    from odps.models.table import TableSchema
    columns = []
    for i, table_column in enumerate(table_schema._columns):
        if select_col_indexes and i not in select_col_indexes:
            continue
        if field_mapping is not None and table_column.name in field_mapping:
            columns.append(Column(name=field_mapping[table_column.name], type=table_column.type))
        else:
            columns.append(table_column)
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
    output_table_schema = TableSchema(columns=columns, partitions=table_schema._partitions)
    return output_table_schema


def get_odps_columns_cnt(table_name):
    from odps import ODPS
    project, input_table_name, _ = parse_table_path(table_name)
    odps = ODPS(access_id=os.getenv('ACCESS_ID'),
                secret_access_key=os.getenv('ACCESS_KEY'),
                project=project,
                endpoint=os.getenv('ODPS_ENDPOINT'))
    if not odps.exist_table(input_table_name, project=project):
        raise ValueError("Table '%s' not exist." % input_table_name)
    input_table = odps.get_table(input_table_name, project=project)
    return len(input_table.table_schema._columns)


def get_odps_columns_name(table_name):
    from odps import ODPS
    project, input_table_name, _ = parse_table_path(table_name)
    odps = ODPS(access_id=os.getenv('ACCESS_ID'),
                secret_access_key=os.getenv('ACCESS_KEY'),
                project=project,
                endpoint=os.getenv('ODPS_ENDPOINT'))
    if not odps.exist_table(input_table_name, project=project):
        raise ValueError("Table '%s' not exist." % input_table_name)
    input_table = odps.get_table(input_table_name, project=project)
    return [_columns.name for _columns in input_table.table_schema._columns]

def get_odps_columns_name_in_tunnel(table_name):
    import common_io
    reader = common_io.table_tunnel.TableReader(table_name)
    col_list = [item[0] for item in reader.get_schema()]
    return col_list


def create_odps_table(
    input_table_path,
    output_table_path,
    select_col_indexes=[],
    generate_num=1,
    output_type=OutputType.Sequence,
    field_mapping=None,
):
    from odps import ODPS
    if input_table_path.startswith("odps://"):
        project, input_table_name, _ = parse_table_path(input_table_path)
        odps = ODPS(access_id=os.getenv('ACCESS_ID'),
                secret_access_key=os.getenv('ACCESS_KEY'),
                project=project,
                endpoint=os.getenv('ODPS_ENDPOINT'))
        if not odps.exist_table(input_table_name, project=project):
            raise ValueError("Table '%s' not exist." % input_table_name)
        input_table = odps.get_table(input_table_name, project=project)
        table_schema = gen_output_table_schema(
            input_table.schema,
            select_col_indexes=select_col_indexes,
            generate_num=generate_num,
            output_type=output_type,
            field_mapping=field_mapping,
        )
        lifecycle=input_table.lifecycle
    elif input_table_path.startswith("ailake://"):
        project, _, __ = parse_table_path(output_table_path)
        odps = ODPS(access_id=os.getenv('ACCESS_ID'),
                secret_access_key=os.getenv('ACCESS_KEY'),
                project=project,
                endpoint=os.getenv('ODPS_ENDPOINT'))
        from .ailake_utils import exist_table, gen_output_table_schema_in_ailake
        if not exist_table(input_table_path):
            raise ValueError("Table '%s' not exist." % input_table_path)
        table_schema = gen_output_table_schema_in_ailake(
            input_table_path,
            select_col_indexes=select_col_indexes,
            generate_num=generate_num,
            output_type=output_type
        )
        # ailake default
        lifecycle=180

    output_table_project, output_table_name, _ = parse_table_path(output_table_path)
    if odps.exist_table(output_table_name, project=output_table_project):
        return
    print(f"create output table {output_table_path}, schema: '{table_schema}' ")
    odps.create_table(name=output_table_name,
                      project=output_table_project,
                      schema=table_schema,
                      if_not_exists=True,
                      lifecycle=lifecycle)


def get_odps_table_length(table_name):
    from common_io.table import TableReader
    with TableReader(table_name, num_threads=0) as reader:
        row_count = reader.get_row_count()
        return row_count
