class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[16, 16, 256, 32]", arg1_1: "f16[16, 16, 256, 32]", arg2_1: "f16[16, 16, 256, 32]"):
        # File: /EuroSys25/attn-compile-print.py:36 in Attention_std, code: scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[3] ** .5)
        expand: "f16[16, 16, 256, 32]" = torch.ops.aten.expand.default(arg1_1, [16, 16, 256, 32]);  arg1_1 = None
        clone: "f16[16, 16, 256, 32]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view: "f16[256, 256, 32]" = torch.ops.aten.reshape.default(clone, [256, 256, 32]);  clone = None
        permute: "f16[16, 16, 32, 256]" = torch.ops.aten.permute.default(arg0_1, [0, 1, 3, 2]);  arg0_1 = None
        expand_1: "f16[16, 16, 32, 256]" = torch.ops.aten.expand.default(permute, [16, 16, 32, 256]);  permute = None
        clone_1: "f16[16, 16, 32, 256]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_1: "f16[256, 32, 256]" = torch.ops.aten.reshape.default(clone_1, [256, 32, 256]);  clone_1 = None
        bmm: "f16[256, 256, 256]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f16[16, 16, 256, 256]" = torch.ops.aten.reshape.default(bmm, [16, 16, 256, 256]);  bmm = None
        div: "f16[16, 16, 256, 256]" = torch.ops.aten.div.Tensor(view_2, 5.656854249492381);  view_2 = None
        
        # File: /EuroSys25/attn-compile-print.py:37 in Attention_std, code: probs = F.softmax(scores, dim=-1)
        convert_element_type_2: "f32[16, 16, 256, 256]" = torch.ops.prims.convert_element_type.default(div, torch.float32);  div = None
        amax: "f32[16, 16, 256, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
        sub: "f32[16, 16, 256, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: "f32[16, 16, 256, 256]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[16, 16, 256, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[16, 16, 256, 256]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_3: "f16[16, 16, 256, 256]" = torch.ops.prims.convert_element_type.default(div_1, torch.float16);  div_1 = None
        
        # File: /EuroSys25/attn-compile-print.py:38 in Attention_std, code: h = torch.matmul(probs, v)
        expand_2: "f16[16, 16, 256, 256]" = torch.ops.aten.expand.default(convert_element_type_3, [16, 16, 256, 256]);  convert_element_type_3 = None
        view_3: "f16[256, 256, 256]" = torch.ops.aten.reshape.default(expand_2, [256, 256, 256]);  expand_2 = None
        expand_3: "f16[16, 16, 256, 32]" = torch.ops.aten.expand.default(arg2_1, [16, 16, 256, 32]);  arg2_1 = None
        clone_2: "f16[16, 16, 256, 32]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_4: "f16[256, 256, 32]" = torch.ops.aten.reshape.default(clone_2, [256, 256, 32]);  clone_2 = None
        bmm_1: "f16[256, 256, 32]" = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
        view_5: "f16[16, 16, 256, 32]" = torch.ops.aten.reshape.default(bmm_1, [16, 16, 256, 32]);  bmm_1 = None
        return (view_5,)
        