from .dataloader import EIPDataloader, FileDataloader, OpenLMDataloader
from .writer import EIPWriter, FileWriter, OdpsTunnelWriter


def create_dataloader_and_writer(
    infer_args, tokenizer, template, processor=None, model_dtype=None, num_return=1, output_type=None
):
    if infer_args.tables:
        input_columns = (
            [infer_args.input_columns[_index] for _index in infer_args.ailake_non_multimodel_data_columns]
            if infer_args.inputs.startswith("ailake://")
            else infer_args.input_columns
        )
        eip_writer = EIPWriter(
            infer_args.inputs, infer_args.outputs, input_columns, num_return, output_type, infer_args.write_buffer_size
        )
        eip_dataloader = EIPDataloader(infer_args, tokenizer, template, processor, model_dtype, output_type)
        return eip_dataloader, eip_writer
    elif infer_args.load_from == "file":
        return FileDataloader(infer_args, tokenizer, template, processor, model_dtype, output_type), FileWriter(
            infer_args.outputs
        )
    elif infer_args.load_from == "openlm":
        return OpenLMDataloader(
            infer_args, tokenizer, template, processor, model_dtype, output_type
        ), OdpsTunnelWriter(infer_args.inputs, infer_args.outputs, infer_args.field_mapping)
    else:
        raise ValueError(f"Unknown load_from: {infer_args.load_from}")
