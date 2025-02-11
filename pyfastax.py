import pyfastx
fasta_list = ["C:\\Users\\srivi\\Desktop\\WSSEF repo\\wssef24-25\\datasets\\sequences\\mutations\\NM_001413459_1.fasta", "C:\\Users\\srivi\\Desktop\\WSSEF repo\\wssef24-25\\datasets\\sequences\\mutations\\NM_001413463_1.fasta", "C:\\Users\\srivi\\Desktop\\WSSEF repo\\wssef24-25\\datasets\\sequences\\mutations\\NM_001413465_1.fasta",
    "C:\\Users\\srivi\\Desktop\\WSSEF repo\\wssef24-25\\datasets\\sequences\\mutations\\NM_001413469_1.fasta",
    "C:\\Users\\srivi\\Desktop\\WSSEF repo\\wssef24-25\\datasets\\sequences\\mutations\\NR_182152_1.fasta",
    "C:\\Users\\srivi\\Desktop\\WSSEF repo\\wssef24-25\\datasets\\sequences\\mutations\\NR_182153_1.fasta",
    "C:\\Users\\srivi\\Desktop\\WSSEF repo\\wssef24-25\\datasets\\sequences\\mutations\\XM_047428319_1.fasta",
    "C:\\Users\\srivi\\Desktop\\WSSEF repo\\wssef24-25\\datasets\\sequences\\mutations\\XM_054371141_1.fasta"]
for file in fasta_list:
    fasta_init = pyfastx.Fastx(file)
    for name, seq in fasta_init:
        print(name, seq)