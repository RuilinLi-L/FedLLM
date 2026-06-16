**FL-LLM**

**Current implementation note (PEFT Leakage)**

The PEFT leakage track is now split into two non-equivalent paths. Existing `attack.py` PEFT evaluation remains DAGER-based and should be reported as a DAGER PEFT span baseline.

`attack_peftleak_image.py --mode vit_adapter` is the PEFTLeak image/adapter reproduction path and reports `attack=peftleak_image_repro`. It uses autograd gradients from a torchvision ViT backbone with a malicious adapter branch. `--mode synthetic_ratio` is kept only as a debug/semantic test path.

`attack_peftleak.py` is the non-DAGER FedLLM PEFT text adaptation and reports `attack=fedllm_peft_text_opt`. It attacks LoRA/IA3 adapter gradients by optimizing dummy input embeddings to match shared PEFT gradients, then decodes embeddings to nearest tokenizer tokens. It should not be reported as the original PEFTLeak reproduction.

For FedLLM PEFT text experiments, the main defense matrix is `none`, `noise`, `dpsgd`, `topk`, `compression`, `soteria`, `mixup`, `lrb`, `lrbprojonly`, and `signed_bottleneck`. The `dager` defense is DAGER-specific and is reported as unsupported for FedLLM PEFT text runs rather than mixed into the main PEFT leakage matrix.

**鎬讳綋鐩爣**

杩欎釜宸ヤ綔鍏虫敞**澶фā鍨嬭缁冭繃绋嬩腑鈥滀腑闂翠俊鎭鑷寸殑鏁版嵁娉勯湶闂鈥?*銆傚凡鏈夌爺绌惰〃鏄庯紝鍦?Transformer/LLM 璁粌涓紝鐗瑰埆鏄湪 FedSGD 杩欑璁惧畾涓嬶紝鏈嶅姟鍣ㄦ槸鏈夊彲鑳介€氳繃瀹㈡埛绔笂浼犵殑姊害鎴栨洿鏂颁俊鎭紝**鍑犱箮瀹屾暣鎭㈠鍑烘湰鍦拌缁冩枃鏈?*鐨勶紝瀹炵幇鐪熷疄璁粌鏁版嵁鐨勬仮澶嶃€?

鍥犳锛屾垜浠殑鐩爣涓嶅彧鏄獙璇佹煇涓€绉嶆敾鍑伙紝鑰屾槸鎯冲洖绛斾竴涓洿鏍稿績鐨勯棶棰橈細**璁粌杩囩▼涓毚闇茬殑杩欎簺涓棿淇℃伅锛屽埌搴曚細娉勯湶澶氬皯鏁版嵁锛熻兘涓嶈兘璁捐涓€绉嶆柟娉曪紝浠庢牴鏈笂鍑忓皯杩欑娉勯湶锛?*

鏁翠綋鎬濊矾鏄紝鍏堝湪涓€涓彲澶嶇幇銆佺畻鍔涘彲鎺х殑璁粌妗嗘灦涓嬶紝鎶婄洰鍓嶅ぇ妯″瀷涓渶涓昏鐨勬暟鎹仮澶嶆敾鍑荤郴缁熸€у鐜板嚭鏉ワ紝鍚屾椂澶嶇幇鐩稿叧闃插尽鏂规硶鐨刡aseline锛涘湪姝ゅ熀纭€涓婏紝鍐嶈璁′竴涓?*灏藉彲鑳介€氱敤鐨勯槻寰℃柟娉?*锛屽幓鍚屾椂鎶戝埗澶氱鏀诲嚮锛屽悓鏃跺敖閲忎笉褰卞搷妯″瀷璁粌鏁堟灉銆傚綋鍓嶉樁娈电殑閲嶇偣涓嶆槸椹笂纭畾闃插尽鏂规硶锛堥渶瑕佷笉鏂皟璇曚互杈惧埌鍙互鍚屾椂闃插尽澶氱鏀诲嚮绫诲瀷骞禨OTA锛夛紝鑰屾槸鍏堟妸鏀诲嚮鍜岃瘎娴嬩綋绯绘惌瀹屾暣銆?

**妗嗘灦浠ュ強鏀诲嚮鏂规硶**

鍦ㄥ疄楠岃瀹氫笂锛屾垜浠互 FedSGD 鐨勮仈閭﹀ぇ妯″瀷璁粌浣滀负涓昏鍦烘櫙锛屽洜涓虹洰鍓嶆渶寮虹殑鏁版嵁鎭㈠鏀诲嚮鍩烘湰閮藉缓绔嬪湪杩欎釜璁惧畾涓娿€傛ā鍨嬭妯℃帶鍒跺湪 7B 浠ヤ笅寮€婧?LLM锛屽苟缁撳悎 PEFT锛堝 LoRA / Adapter锛?鏉ラ檷浣庣畻鍔涘紑閿€銆傝缁冩鏋舵柟闈富瑕佸熀浜庡凡鏈夌殑鏀诲嚮妗嗘灦鏉ュ疄鐜般€?

鍦ㄦ敾鍑婚€夋嫨涓婏紝鎴戜滑涓嶅啀绠€鍗曞爢鍙犳墍鏈夊凡鏈夋柟娉曪紝鑰屾槸鎸夌収**涓棿淇℃伅濡備綍娉勯湶鏁版嵁**鏉ヨ繘琛屽垎绫伙紝骞堕噸鐐硅鐩栦笁绫绘渶鏈変唬琛ㄦ€х殑鏀诲嚮锛?

绗竴绫绘槸 **鍩轰簬姊害鐨勬仮澶嶆敾鍑伙紙Gradient Inversion锛?*锛屽寘鎷?DAGER 杩欑被鍙互鍦ㄥぇ妯″瀷涓嚑涔庣簿纭仮澶嶆枃鏈殑鏀诲嚮锛屼互鍙婄被浼?LAMP 杩欑被杈冩棭鐨勬枃鏈搴︽硠闇叉柟娉曘€傝繖绫绘敾鍑讳富瑕佸埄鐢ㄦ搴︿笌杈撳叆鏁版嵁涔嬮棿鐨勭洿鎺ユ槧灏勫叧绯汇€?

**浠ｈ〃鏂囩珷锛欴AGER: Exact Gradient Inversion for Large Language Models锛堜富瑕侊級LAMP锛堟瑕侊級**

绗簩绫绘槸 **PEFT 鐩稿叧鐨勬暟鎹硠闇诧紙PEFT Leakage锛?*锛屽寘鎷?PEFTLeak 鍜?ReCIT 绛夊伐浣溿€傝繖绫绘敾鍑昏鏄庯紝鍗充娇鍙叡浜?LoRA 鎴?Adapter 杩欐牱鐨勮交閲忔洿鏂帮紝浠嶇劧鍙兘鎭㈠鍑鸿缁冩暟鎹紝鍥犳瀵瑰疄闄呭ぇ妯″瀷璁粌鏇村叿鐜板疄鎰忎箟銆?

**浠ｈ〃鏂囩珷锛欸radient Inversion Attacks on Parameter-Efficient Fine-Tuning锛堜富瑕侊級ReCIT锛堟棤浠ｇ爜锛屾瑕侊級**

绗笁绫绘槸 **灞傜骇鎴栧眬閮ㄤ俊鎭硠闇诧紙Layer-level Leakage锛?*锛屽嵆鍙埄鐢ㄩ儴鍒嗗眰鐨勬搴︽垨鏇存柊淇℃伅锛屽氨鍙互鎭㈠鏁版嵁銆傝繖绫绘敾鍑昏鏄庯紝鏁版嵁娉勯湶涓嶄竴瀹氫緷璧栧畬鏁存搴︼紝灞€閮ㄤ俊鎭悓鏍峰彲鑳芥槸鍗遍櫓鐨勩€?

**浠ｈ〃鏂囩珷锛歋eeing the Forest through the Trees: Data Leakage from Partial Transformer Gradients**

杩欎笁绫绘敾鍑昏鐩栦簡褰撳墠澶фā鍨嬭缁冧腑涓昏鐨勬硠闇茶矾寰勶紝涔熸瀯鎴愪簡鏈伐浣滅殑鏍稿績濞佽儊妯″瀷锛?*瀹為獙涓紝灏卞彲浠ョ敤涓婇潰鐨勪笁涓富瑕佹枃绔犱綔涓烘垜浠殑鏀诲嚮鏂规硶锛岄兘鏄湁瀹屾暣婧愮爜鐨?*锛夈€?

| 璁烘枃 | 绫诲埆 | 鏍稿績鏂规硶 | 浼樼偣 | 缂虹偣/灞€闄?| 鏄惁寮€婧?|
| --- | --- | --- | --- | --- | --- |
| DAGER: Exact Gradient Inversion for Large Language Models | Gradient Inversion | 灏嗘搴﹀弽婕斿缓妯′负绮剧‘浼樺寲闂锛岀洿鎺ユ仮澶峵oken搴忓垪 | 褰撳墠鏈€寮烘敾鍑讳箣涓€锛屽彲杩戜箮绮剧‘鎭㈠鏂囨湰 | 渚濊禆瀹屾暣姊害锛岃绠楀紑閿€澶?| 鉁?|
| LAMP: Extracting Text from Gradients with Language Model Priors | Gradient Inversion | 缁撳悎璇█妯″瀷鍏堥獙杈呭姪姊害鍙嶆紨 | 鐩告瘮鏃╂湡鏂规硶鎭㈠鏁堟灉鏇村ソ锛屾€濊矾閫氱敤 | 鎭㈠绮惧害涓嶅DAGER锛屽亸鏃╂湡鏂规硶 | 鉁?|
| Gradient Inversion Attacks on Parameter-Efficient Fine-Tuning | PEFT Leakage | 浠嶭oRA/Adapter鏇存柊涓仮澶嶈缁冩暟鎹?| 璇佹槑杞婚噺鏇存柊鍚屾牱娉勯湶鏁版嵁锛岀幇瀹炴剰涔夊己 | 渚濊禆鐗瑰畾PEFT缁撴瀯 | 鉁?|
| ReCIT: Reconstruction of Training Data from Incremental Training | PEFT Leakage | 鍒╃敤澧為噺璁粌鏇存柊杩涜鏁版嵁閲嶆瀯 | 涓嶄緷璧栧畬鏁存搴︼紝鏇磋创杩戝疄闄呰缁?| 澶嶇幇闅惧害楂橈紝鏃犲叕寮€浠ｇ爜 | 鉂?|
| Seeing the Forest through the Trees: Data Leakage from Partial Transformer Gradients | Layer-level Leakage | 浠呭埄鐢ㄩ儴鍒嗗眰姊害鎭㈠鏁版嵁 | 璇佹槑灞€閮ㄤ俊鎭篃浼氭硠闇诧紝濞佽儊鏇村箍娉?| 鎭㈠绮惧害浣庝簬鍏ㄦ搴︽敾鍑?| 鉁?|

鍙弬鑰冪患杩帮細Analysis of Privacy Leakage in Federated Large Language Models

LLM in the middle: A systematic review of threats and mitigations to real-world LLM-based systems

**Baselines**

| 璁烘枃 | 绫诲埆 | 鏍稿績鏂规硶 | 浼樼偣 | 缂虹偣/灞€闄?| 鏄惁寮€婧?|
| --- | --- | --- | --- | --- | --- |
| Deep Learning with Differential Privacy (DP-SGD) | Defense (DP) | 鍘熷鏂规硶閫氳繃閫愭牱鏈鍓€佸姞鍣０鍜?accountant 鎶ュ憡 privacy cost | 鐞嗚鏈€瀹屽杽锛屾爣鍑嗘柟娉?| 瀵规ā鍨嬫€ц兘褰卞搷鏄庢樉锛涙湰浠撳簱 LoRA eval 鍙疄鐜?DP-SGD-style 姊害鍙樻崲锛屼笉鎻愪緵 formal DP guarantee | 鉁?|
| Soteria: Provable Defense against Privacy Leakage in Federated Learning | Defense (Representation) | 瀵逛腑闂磋〃绀鸿繘琛屾壈鍔紝闄嶄綆鍙仮澶嶆€?| 涓撻棬閽堝姊害娉勯湶璁捐 | 瀵瑰鏉傛ā鍨嬬ǔ瀹氭€ф湁闄愶紱鏈粨搴?LoRA eval 鏄?Soteria-style representation masking锛屼笉鏄畬鏁磋缁冩湡澶嶇幇 | 鉁?|
| Gradient Compression for Communication-Efficient FL | Defense (Compression) | 瀵规搴﹁繘琛岀█鐤忓寲鎴栭噺鍖栧帇缂?| 闄嶄綆閫氫俊鍚屾椂鍑忓皯淇℃伅閲?| 涓嶆槸涓撲负闅愮璁捐锛岄槻鎶ゆ湁闄?| 鉁?|
| Top-k Gradient Sparsification | Defense (Compression) | 浠呬繚鐣欐搴︿腑鏈€澶涓垎閲?| 绠€鍗曢珮鏁堬紝鏄撳疄鐜?| 淇℃伅浠嶅彲鑳借鎭㈠ | 鉁?|
| MixUp / Manifold MixUp-style Defense | Defense (Data/Representation-level) | 閫氳繃杈撳叆鎴栭殣钘忚〃绀烘贩鍚堥檷浣庢牱鏈彲璇嗗埆鎬?| 瀹炵幇绠€鍗曪紝鏃犻渶鏀规ā鍨嬬粨鏋?| 瀵规敾鍑荤殑閽堝鎬ц緝寮憋紱鏈粨搴撳疄鐜颁负 representation-level manifold MixUp-style | 鉁?|
| Noise Injection on Gradients | Defense (Noise) | 鐩存帴瀵规搴︽坊鍔犻殢鏈烘壈鍔?| 绠€鍗曠洿鎺ワ紝鏄撲簬鎺у埗寮哄害 | 闅愮-鏁堢敤鏉冭　闅捐皟 | 鉁?|
## 杩欓噷寰€涓嬩笉鏄師濮媔dea鐨勫唴瀹癸紝鏄悗闈i鐢熸垚鐨?
**瀵瑰綋鍓?baselines 鐨勫垽鏂?*

濡傛灉缁撳悎鏈枃鐨勪笁绫绘牳蹇冨▉鑳佹ā鍨嬫潵鐪嬶紝鐜版湁 baselines 鐨勪环鍊兼洿澶氭槸鈥滄彁渚涙湁浠ｈ〃鎬х殑瀵圭収缁勨€濓紝鑰屼笉鏄洿鎺ユ垚涓烘渶缁堟柟娉曘€傛洿鍏蜂綋鍦拌锛?

| baseline | 涓昏浼樺娍 | 涓昏涓嶈冻 | 鏇撮€傚悎鎵紨鐨勮鑹?|
| --- | --- | --- | --- |
| `noise` | 鏈€閫氱敤銆佹渶渚垮疁銆佸疄鐜版渶绠€鍗曪紝閫傚悎浣滀负鏈€灏忓共棰勫熀绾?| 鍙槸鍦ㄨ緭鍑虹鍋氶殢鏈烘壈鍔紝涓嶆敼鍙樻硠闇茬粨鏋勶紱寰€寰€闇€瑕佽緝澶у櫔澹版墠鏈夋晥锛屾晥鐢ㄤ笅闄嶅揩 | 鏈€鍩虹鐨?sanity-check baseline |
| `dpsgd` | 鍘熷 DP-SGD 鏄湁鏄庣‘宸垎闅愮璇箟鐨勬爣鍑?baseline锛岃鏂囪鏈嶅姏寮?| 褰撳墠 LoRA eval 鍙仛閫愭牱鏈鍓€佸钩鍧囥€佸姞楂樻柉鍣０锛屾病鏈夎缁冩湡鎺ュ叆鍜?privacy accountant锛涗笉鑳藉０绉?formal DP guarantee | DP-SGD-style attack-time baseline / clipping + Gaussian noise baseline |
| `topk` | 璁＄畻涓庨€氫俊鎴愭湰浣庯紝瀹规槗鍦ㄥぇ妯″瀷涓婅窇澶ц妯?sweep | 淇濈暀涓嬫潵鐨勫線寰€姝ｆ槸鏈€鏄捐憲銆佹渶鍙兘鎼哄甫鍏抽敭淇℃伅鐨勫垎閲忥紱涓嶆槸涓洪殣绉佽璁?| 閫氫俊鍘嬬缉瀵圭収缁?|
| `compression` | 鍏奸【閫氫俊鏁堢巼涓庝竴瀹氫俊鎭崯澶憋紝宸ョ▼涓婂鏄撴帴鍏?| 鏇村鏄湪鈥滃皯浼犲灏戔€濊€屼笉鏄€滃皯娉勯湶澶氬皯鈥濅笂鏈夋晥锛涘寮烘敾鍑荤殑閽堝鎬ф湁闄?| 鍘嬬缉绫?baseline |
| `soteria` | 鐩存帴瑙﹀強 representation leakage锛屾湰璐ㄤ笂姣旂函姊害鍚庡鐞嗘洿鎺ヨ繎娉勯湶鏍瑰洜 | 褰撳墠 LoRA eval 鏄?classifier-input representation masking 鍚庨噸绠?adapter 姊害锛涗笉鏄師濮?Soteria 鐨勫畬鏁村鐜帮紝涔熶笉鏄?LoRA 璁粌鏈熼槻寰?| Soteria-style representation masking baseline |
| `mixup` | 瀵硅缁冩晥鐢ㄩ€氬父杈冨弸濂斤紝涔熷彲鑳介檷浣庢牱鏈敮涓€鎬?| 褰撳墠瀹炵幇娣峰悎 hidden/classifier-input representation 鍜屾爣绛撅紝涓嶆槸鍘熷 input-level MixUp锛涘湪灏?batch 鐨?FL 璁惧畾閲岋紝娣峰悎鍚庣殑姊害浠嶅彲鑳芥硠闇茶緝寮轰俊鎭?| Manifold MixUp-style representation baseline |
| `lrb`锛堝綋鍓嶇増鏈級 | 鏈€鎺ヨ繎鏈枃鈥滈€氱敤闃插尽鈥濈洰鏍囷紝鏄惧紡鍒╃敤灞傜骇宸紓鍜屾仮澶嶇摱棰堟潵鍋氱粨鏋勫寲鎶戝埗 | 鐩墠杩樻槸鍚彂寮?v1锛氭晱鎰熷眰璇嗗埆涓昏闈犺鍒欙紝鎶曞奖鍩哄簳杈冨浐瀹氾紝涔熻繕娌℃湁 forward-side 鐨勮〃绀虹害鏉?| 鏈€鏈夊笇鏈涘彂灞曟垚涓绘柟娉曠殑鍘熷瀷 |

鎬荤粨璧锋潵锛宍noise / dpsgd` 鏇村儚鈥滈€氱敤鎵板姩绯烩€濆熀绾匡紝鍏朵腑 `dpsgd` 鍦ㄦ湰浠撳簱 LoRA eval 涓簲鏄庣‘鍐欐垚 DP-SGD-style锛沗topk / compression` 鏇村儚鈥滈€氫俊鍘嬬缉绯烩€濆熀绾匡紱`soteria` 鏄?Soteria-style 琛ㄧず鎵板姩鍩虹嚎锛沗mixup` 鏄?manifold MixUp-style 琛ㄧず娣峰悎鍩虹嚎銆傜湡姝ｅ拰鏈枃鐩爣鏈€涓€鑷寸殑锛屾槸浠?`lrb` 缁х画寰€鍓嶆帹锛屽洜涓哄畠寮€濮嬫樉寮忛拡瀵光€滃摢浜涗腑闂翠俊鎭洿鍙仮澶嶁€濊繖涓棶棰樻湰韬紝鑰屼笉鍙槸绮楁毚鍑忓皯鏁板€肩簿搴︺€?## 杩欓噷寰€涓婁笉鏄師濮媔dea鐨勫唴瀹癸紝鏄悗闈i鐢熸垚鐨勶紝涓嬮潰鐨勬槸鍘焛dea鍐呭
**闃插尽鏂规硶璁捐**

鍦ㄩ槻寰¤璁′笂锛屾垜浠殑鐩爣涓嶆槸閽堝鏌愪竴涓叿浣撴敾鍑诲幓鍋氫紭鍖栵紝鑰屾槸灏濊瘯璁捐涓€绉?*閫氱敤鐨勯槻寰℃満鍒?*锛岃兘澶熸暣浣撻檷浣庘€滀腑闂存洿鏂颁俊鎭?鈫?鍘熷鏁版嵁鈥濈殑娉勯湶鑳藉姏銆傛崲鍙ヨ瘽璇达紝鎴戜滑甯屾湜闃插尽鐨勪笉鏄煇涓敾鍑荤畻娉曪紝鑰屾槸**鏁版嵁浠庢搴︽垨鏇存柊涓鎭㈠鍑烘潵杩欎欢浜嬫湰韬紝鍦ㄥ疄楠屽眰闈㈠氨鏄紝鍚屾椂鍦ㄤ笂闈㈢殑涓夌绫诲瀷鏀诲嚮褰撲腑鐢ㄦ垜浠殑闃插尽绠楁硶杈惧埌鏈€浼樼殑闅愮-鏁堢敤鏉冭　**銆?

浠庣洿瑙変笂鐪嬶紝澶фā鍨嬭缁冧腑娑夊強绂绘暎 token 鍒拌繛缁〃绀虹殑鏄犲皠锛岃繖涓€杩囩▼鏈韩灏辨彁渚涗簡寰堝鍙互骞查鐨勭┖闂达紝姣斿鍦ㄦ洿鏂颁俊鎭腑寮曞叆鎵板姩銆佹敼鍙樿〃绀烘柟寮忥紝鎴栬€呴檺鍒朵俊鎭〃杈捐兘鍔涚瓑銆傚洜姝ゅ悗缁彲浠ヤ粠澶氫釜鏂瑰悜鎺㈢储闃插尽鏂规硶锛屼緥濡傚姊害缁撴瀯杩涜淇敼銆佸鏇存柊淇℃伅杩涜鎵板姩鎴栧帇缂╋紝鎴栬€呭湪璁粌杩囩▼涓檺鍒朵俊鎭硠闇层€?

褰撳墠闃舵涓嶄細鎻愬墠鍥哄畾鏌愪竴绉嶅叿浣撴柟妗堬紝鑰屾槸鍏堝湪缁熶竴鐨勫疄楠屾鏋朵笅楠岃瘉涓€涓牳蹇冮棶棰橈細**鏄惁瀛樺湪涓€绉嶆柟娉曪紝鍙互鍦ㄥ熀鏈笉褰卞搷璁粌鏁堟灉鐨勫墠鎻愪笅锛屽悓鏃堕檷浣庡绫绘暟鎹仮澶嶆敾鍑荤殑鎴愬姛鐜?*銆傚湪杩欎釜鍩虹涓婏紝鍐嶉€愭鏀舵暃鍒板叿浣撶殑闃插尽璁捐銆?*锛堣繖涓槸褰撳墠闇€瑕佽€冭檻鐨勯噸鐐癸紝涔熷氨鏄鐜版敾鍑绘柟娉曠殑鍩虹涓婏紝鍚屾椂灏濊瘯澶嶇幇鍏惰嚜甯︽垨鑷繁鍔犲叆涓€浜沚aseline锛岀劧鍚庡啀璁捐鎴戜滑鑷繁鐨勭畻娉曪紝鐪嬭兘涓嶈兘鎵撹繃杩欐墦杩囪繖浜沚aseline锛?*
## 杩欓噷寰€涓嬩笉鏄師濮媔dea鐨勫唴瀹癸紝鏄悗闈i鐢熸垚鐨?
**寤鸿鐨勫叿浣撴柟娉曡璁★細LRB-v2 / HLRB**

濡傛灉娌跨潃鏈枃鐩爣缁х画浼樺寲锛屾垜涓嶅缓璁畬鍏ㄦ帹缈?`lrb`锛岃€屾槸寤鸿鎶婂畠鍗囩骇鎴愪竴涓洿瀹屾暣鐨勪袱闃舵閫氱敤闃插尽锛?*HLRB锛圚ierarchical Layer-wise Recoverability Bottleneck锛?*銆傚畠鍙互鐞嗚В涓?`lrb` 鐨勭爺绌剁増 v2銆?

鏍稿績鎬濇兂涓嶆槸鍘烩€滈獥杩囨煇涓敾鍑烩€濓紝鑰屾槸涓诲姩闄愬埗璁粌杩囩▼涓渶瀹规槗琚仮澶嶇殑閭ｉ儴鍒嗕俊鎭紝璁╁叡浜洿鏂板彧淇濈暀瀹屾垚浠诲姟鎵€蹇呴渶銆佷絾涓嶅埄浜庢仮澶嶅師濮嬫牱鏈殑鎴愬垎銆?

HLRB 鍙互鍒嗘垚涓や釜灞傞潰锛?

1. **forward-side representation bottleneck**

- 鍦ㄧ湡姝ｈ涓嬫父澶撮儴鎴?PEFT 妯″潡娑堣垂鐨勮〃绀轰笂鍔犲叆杞婚噺鐡堕锛岃€屼笉鏄彧鍦ㄦ渶缁堟搴︿笂鍚庡鐞嗐€?
- 瀵?`seq_class` 浠诲姟锛屽彲浠ュ儚 `Soteria` 涓€鏍蜂粠鈥滃垎绫诲ご鐪熸浣跨敤鐨勮〃绀衡€濆嚭鍙戯紝浣嗕笉鍋氫竴娆℃€х‖鍓灊锛岃€屾槸鍋氭洿骞虫粦鐨勪綆缁存姇褰便€侀殢鏈哄瓙绌洪棿鎺╃爜鎴栧彈鎺?dropout銆?
- 瀵?PEFT 鍦烘櫙锛屽彲浠ユ妸 bottleneck 鏀惧湪 LoRA/Adapter 鐨勮緭鍏ユ垨鐡堕婵€娲讳笂锛岀洿鎺ラ檺鍒跺彲鎭㈠淇℃伅杩涘叆鍙缁冩ā鍧椼€?

2. **backward-side recoverability bottleneck**

- 瀵规瘡涓€灞傚叡浜搴?`G_l`锛屽厛浼拌涓€涓硠闇叉晱鎰熷害 `s_l`锛屽啀鎸夋晱鎰熷害鍐冲畾璇ュ眰鐨勪繚鐣欐瘮渚嬨€佽鍓己搴﹀拰鍣０寮哄害銆?
- 涓嶆槸鍍忓綋鍓?`lrb` v1 閭ｆ牱涓昏渚濊禆灞傚悕瑙勫垯锛岃€屾槸寮曞叆涓€涓?*鏍″噯姝ラ**锛氬湪鍏紑鏁版嵁鎴?warmup batch 涓婄粺璁℃瘡灞傜殑姊害鑳介噺銆佽氨闆嗕腑搴︺€佹棭灞?embedding 閲嶈鎬с€佷互鍙婂鏀诲嚮 proxy 鐨勫彲鎭㈠鎬с€?
- 瀵逛簬姣忓眰姊害锛屽仛鍒嗚В  
  `G_l = P_l(G_l) + R_l(G_l)`  
  鍏朵腑 `P_l` 鏄厑璁镐繚鐣欑殑浣庢仮澶嶆€у叕鍏卞瓙绌洪棿锛宍R_l` 鏄洿鍙兘娉勯湶鏍锋湰缁嗚妭鐨勬畫宸瓙绌洪棿銆?
- 鏈€缁堝叡浜殑鏄? 
  `\tilde{G}_l = clip(P_l(G_l)) + \xi_l`锛屽叾涓?`\xi_l` 涓昏鍔犲湪娈嬪樊鏂瑰悜鎴栦笌淇濈暀瀛愮┖闂存浜ょ殑鏂瑰悜涓娿€? 
  杩欐牱鍋氱殑鐩磋鏄細灏介噺淇濈暀浠诲姟鐩稿叧鐨勪綆棰?绋冲畾缁撴瀯锛屼紭鍏堢牬鍧忔槗鎭㈠鐨勯珮棰?灞€閮?鏍锋湰鐗瑰紓淇℃伅銆?

鐩稿褰撳墠 `lrb` v1锛屾垜寤鸿鐨勫叧閿崌绾х偣鏄細

- **鏁忔劅搴︿及璁′粠瑙勫垯鏀逛负鏍″噯**  
  褰撳墠鐗堟湰涓昏渚濊禆鈥渆mbedding 鍜屽墠鍑犲眰鏇存晱鎰熲€濈殑鍚彂寮忥紝杩欎釜鏂瑰悜鏄鐨勶紝浣嗚繕涓嶅绋炽€傛洿濂界殑鍋氭硶鏄负姣忕妯″瀷鍜岃缁冩柟寮忛鍏堣窇涓€娆?calibration锛屽緱鍒?layer-wise sensitivity profile銆?

- **鎶曞奖鍩哄簳浠庡浐瀹?pooling 鏀逛负鍏叡浣庢仮澶嶆€у瓙绌洪棿**  
  褰撳墠鐨?adaptive pooling 寰堜究瀹滐紝浣嗗甫鏈夎緝寮哄潗鏍囩郴鍋囪銆傛洿绋崇殑鏂规鏄敤鍥哄畾闅忔満姝ｄ氦鍩恒€丠adamard 椋庢牸鍩猴紝鎴栬€呯敱鍏紑鏁版嵁浼拌鍑虹殑浣庣З鍏叡瀛愮┖闂存潵鍋氭姇褰便€?

- **鍣０鍔犲湪鈥滆涓㈠純鐨勬柟鍚戔€濊€屼笉鏄钩鍧囦贡鍔?*  
  杩欐瘮鍏ㄧ┖闂村姞鍣洿鑺傜渷鏁堢敤棰勭畻锛屼篃鏇寸鍚堚€滈噸鐐规墦鏂仮澶嶈矾寰勨€濈殑璁捐鐩爣銆?

- **鍏煎 full fine-tuning 涓?PEFT**  
  瀵?full 妯″瀷锛岄噸鐐逛繚鎶?embedding 鍜屾棭灞?attention 鐩稿叧鍙傛暟锛涘 LoRA/Adapter锛岄噸鐐逛繚鎶?bottleneck 杈撳叆杈撳嚭鍜屽墠鍑犲眰閫傞厤鍣紝鑰屼笉鏄彧鎶?PEFT 鐪嬫垚鈥滃皬鍙傛暟閲忔墍浠ュ簲璇ユ洿瀹夊叏鈥濄€?

- **鎶婃柟娉曠洰鏍囧啓鎴愨€滄渶灏忓寲 recoverability锛岃€屼笉鏄渶澶у寲鎵板姩鈥?*  
  杩欑偣寰堥噸瑕併€傜爺绌跺彊浜嬩笂锛孒LRB 鐨勭洰鏍囦笉鏄妸姊害寮勫緱瓒婁贡瓒婂ソ锛岃€屾槸鍦ㄧ粰瀹氭晥鐢ㄦ崯澶遍绠椾笅锛屾渶澶ч檺搴﹀帇浣庤法鏀诲嚮闈㈢殑 recoverability銆?

**涓轰粈涔堣繖涓柟鍚戞瘮鍗曚竴 baseline 鏇撮€傚悎浣滀负涓绘柟娉?*

- 瀹冨拰鏈枃濞佽儊妯″瀷鏇翠竴鑷淬€傛湰鏂囦笉鏄彧鎵?DAGER锛屼篃涓嶆槸鍙墦鏌愪釜 PEFT 鏀诲嚮锛岃€屾槸瑕佸悓鏃堕潰瀵?full-gradient銆丳EFT銆乸artial-gradient 涓夌被娉勯湶銆?
- 瀹冩瘮 `noise / DP-SGD-style` 鏇寸粨鏋勫寲锛屾瘮 Soteria-style 琛ㄧず鍓灊鏇撮€氱敤锛屾瘮 `topk/compression` 鏇翠互闅愮涓轰腑蹇冦€?- 瀹冧繚鐣欎簡 `lrb` 褰撳墠瀹炵幇鐨勫伐绋嬩紭鍔匡細鍙互鍏堜粠 attack-time transform 鍋?v1/v2锛屽姣旂粨鏋滃嚭鏉ュ悗锛屽啀鍐冲畾鏄惁鎺ㄨ繘鍒拌缁冩椂鐗堟湰銆?

**鐜伴樁娈垫渶鎺ㄨ崘鐨勭爺绌惰矾绾?*

濡傛灉鍙€夋嫨涓€鏉′富绾匡紝鎴戝缓璁細

- 淇濈暀 `noise / dpsgd / topk / compression / soteria / mixup` 浣滀负 baseline 濂椾欢锛屽叾涓?LoRA eval 涓?`dpsgd / soteria / mixup` 鍒嗗埆鎸?DP-SGD-style銆丼oteria-style銆乵anifold MixUp-style 鎶ュ憡銆?- 鎶?`lrb` 鏄庣‘瀹氫綅涓衡€滄垜浠殑涓绘柟娉曞師鍨嬧€濓紝鍚庣画鍗囩骇涓?`LRB-v2 / HLRB`銆?
- 鍏堝仛 `post-gradient HLRB`锛岄獙璇佹槸鍚﹁兘鍚屾椂鍘嬩綆涓夌被鏀诲嚮銆?
- 濡傛灉缁撴灉鎴愮珛锛屽啀杩涗竴姝ュ疄鐜?`representation-side HLRB`锛屾妸 forward bottleneck 涔熺撼鍏ャ€?

杩欐牱鍋氱殑濂藉鏄細鐮旂┒涓荤嚎娓呮櫚锛屽疄楠屽鐓у厖鍒嗭紝鑰屼笖鏂规硶婕旇繘璺緞鑷劧锛屼笉闇€瑕佺獊鐒朵粠 baseline 璺冲埌涓€涓畬鍏ㄤ笉鍚屻€侀毦浠ヨВ閲婄殑鏂版柟娉曘€?
## 杩欓噷寰€涓婁笉鏄師濮媔dea鐨勫唴瀹癸紝鏄悗闈i鐢熸垚鐨勶紝涓嬮潰鐨勬槸鍘焛dea鍐呭
**宸ヤ綔鎬ц川鍒ゆ柇**

鏁翠綋鏉ョ湅锛岃繖椤瑰伐浣滄槸涓€涓互**瀹炶返鍜岄棶棰樹负瀵煎悜鐨勭爺绌跺瀷宸ヤ綔**锛岃€屼笉鏄亸鐞嗚鐮旂┒銆備富瑕佸叧娉ㄧ偣鍦ㄤ簬鐪熷疄鑱旈偊澶фā鍨嬭缁冨満鏅笅闅愮椋庨櫓鏄惁瀛樺湪銆侀闄╂湁澶氫弗閲嶏紝浠ュ強鍦ㄧ幇瀹炵畻鍔涘拰绯荤粺绾︽潫涓嬭兘鍚︽湁鏁堢紦瑙ｈ繖浜涢闄┿€傜悊璁哄垎鏋愭洿澶氫綔涓哄瀹為獙鐜拌薄鍜岄槻寰℃満鍒剁殑瑙ｉ噴鍜屾敮鎾戯紝鑰屼笉鏄牳蹇冪洰鏍囥€?
**褰撳墠瀹為獙钀藉湴鍙ｅ緞锛?026-05锛?*

褰撳墠浠ｇ爜宸茬粡褰㈡垚涓夋潯鍙墽琛屼富绾匡細

1. **full-gradient DAGER baseline / defense**
   鐢?`scripts/defense_baselines.sh` 鍜?`scripts/lrb_ablation.sh` 璺?`none / topk / compression / lrb` 绛?full-gradient 鏀诲嚮璇勬祴涓?LRB 娑堣瀺銆傛渶鏂版秷铻嶇粨璁烘槸锛歚proj_only@0.5` 鏄綋鍓?`SST2 + GPT2 + batch=2` full-gradient DAGER 涓嬫渶閲嶈鐨?LRB 涓诲€欓€夛紝`full_lrb@0.5` 淇濈暀涓哄畬鏁村己闃插尽瀵圭収銆?
2. **LoRA / PEFT eval-first 鏀诲嚮闈?*
   鍏堢敤 `train.py --train_method lora` 璁粌骞朵繚瀛樼湡瀹炵殑 LoRA checkpoint锛屼緥濡?`./models/gpt2_sst2_lora_r16/final_adapter`锛涘啀鐢?`scripts/peft_eval.sh` 鎴?`scripts/peft_baselines.sh` 璇勪及 LoRA 鏇存柊涓嬬殑 `none / proj_only / proj_clip / full_lrb / topk@0.1 / compression@8`锛屼互鍙?DP-SGD-style / Soteria-style / manifold MixUp-style 杩欎笁绫?eval-only baseline銆傚綋鍓?eval 鏀寔 PEFT adapter 鐩綍锛屼篃鍏煎鏈湴 `.pt/.pth` LoRA `state_dict`銆?
3. **涓嬩竴闃舵娉涘寲楠岃瘉**
   LoRA/PEFT 鍜?partial-gradient 鏄獙璇?LRB 鏄惁鍏锋湁璺ㄦ敾鍑婚潰缁撴瀯鎬т环鍊肩殑鍏抽敭銆傛枃妗ｆ墽琛屽叆鍙ｄ互 `docs/PEFT_EVAL.md` 鍜?`docs/LRB_ABLATION_ANALYSIS_20260503.md` 鐨勭 13 鑺備负鍑嗭紝閬垮厤缁х画浣跨敤 `path/to/lora_checkpoint.pt` 杩欑被鍗犱綅绗﹁矾寰勩€?

