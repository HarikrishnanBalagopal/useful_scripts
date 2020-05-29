rm -rf misc/
rm -rf PQG-pytorch/
git clone --single-branch --branch orig-code https://github.com/dev-chauhan/PQG-pytorch.git
cp -r PQG-pytorch/misc .

mkdir bucket1/
gcsfuse --implicit-dirs iitk-thesis-data bucket1/
rm -rf datasets/
mkdir datasets/
cp bucket1/datasets/* datasets/
cp -r bucket1/datasets/quora_paraphrase_paper_authors_results/ datasets/
cp -r bucket1/datasets/quora_paraphrase/ datasets/
cp datasets/quora_paraphrase_paper_authors_results/faster_text_gen_v1.pth .
cp datasets/quora_paraphrase_paper_authors_results/faster_text_dis_v1.pth .

conda install -y -c conda-forge easydict
echo 'DONE!'
