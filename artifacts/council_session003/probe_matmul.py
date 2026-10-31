import os, time, json, statistics
os.environ.setdefault('SDAA_QUIET','1')
import torch
info={'torch':torch.__version__}
try:
    import torch_sdaa
    info['torch_sdaa']=getattr(torch_sdaa,'__version__','?')
    info['sdaa_available']=torch.sdaa.is_available()
    info['device_count']=torch.sdaa.device_count() if hasattr(torch.sdaa,'device_count') else None
except Exception as e:
    info['torch_sdaa_error']=repr(e)
res={}
if info.get('sdaa_available'):
    dev='sdaa:0'
    a=torch.randn(1200,1200,device=dev); b=torch.randn(1200,1200,device=dev)
    for _ in range(10): c=a@b
    torch.sdaa.synchronize()
    ts=[]
    for _ in range(50):
        t=time.perf_counter(); c=a@b; torch.sdaa.synchronize(); ts.append((time.perf_counter()-t)*1e3)
    res={'mean_ms':statistics.mean(ts),'median_ms':statistics.median(ts),'min_ms':min(ts),'max_ms':max(ts),
         'cv_pct':statistics.pstdev(ts)/statistics.mean(ts)*100,'first5':[round(x,2) for x in ts[:5]],'last5':[round(x,2) for x in ts[-5:]]}
print(json.dumps({'info':info,'matmul':res},ensure_ascii=False))
