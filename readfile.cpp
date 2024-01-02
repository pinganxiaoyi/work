
#include<iostream>
#include<fstream>

using namespace std;

void readFile(){

    TChain *ch_mc = new TChain("events");
    ch_mc->Add("/herdfs/user/herdprd/testMC/1test/v2022a-fit+calo/gamma/top10cmx10cmIso/50/gamma-top10cmx10cmVert-sim.43265635.*.root");
   //try batch    
    ofstream outfile("output_Count.txt")
    TObject *obj = ch_mc->GetListOfFiles();
    TIter next(obj);
    TObjString *NameObj;
    while ((NameObj = (TObjString*)next())){
        TString filename = NameObj->GetString();
        cout << "Processing file: " << filename << endl;
        TChain *singleFileChain = new TChain("events");
        singleFileChain->Add(filename);
        TTreeReader reader_mc(singleFileChain);


    //mc particle
    TTreeReaderArray<int> mcparts_trackID(reader_mc, "mcparts.trackID");
    TTreeReaderArray<int> mcparts_parentID(reader_mc, "mcparts.parentID");
    TTreeReaderArray<int> mcparts_pdgID(reader_mc, "mcparts.pdgID");
    TTreeReaderArray<float> mcparts_vertex_x(reader_mc, "mcparts.vertex.x");
    TTreeReaderArray<float> mcparts_vertex_y(reader_mc, "mcparts.vertex.y");
    TTreeReaderArray<float> mcparts_vertex_z(reader_mc, "mcparts.vertex.z");
    TTreeReaderArray<float> mcparts_momentum_x(reader_mc, "mcparts.momentum.x");
    TTreeReaderArray<float> mcparts_momentum_y(reader_mc, "mcparts.momentum.y");
    TTreeReaderArray<float> mcparts_momentum_z(reader_mc, "mcparts.momentum.z");

   //fit hits
    TTreeReaderArray<int>   hit_trackID(reader_mc, "fithits.trackID");
    TTreeReaderArray<float> hit_posX(reader_mc, "fithits.pos.x");
    TTreeReaderArray<float> hit_posY(reader_mc, "fithits.pos.y");
    TTreeReaderArray<float> hit_posZ(reader_mc, "fithits.pos.z");
    TTreeReaderArray<float> hit_edep(reader_mc, "fithits.edep");
    TTreeReaderArray<int> hit_pdgID(reader_mc, "fithits.pdgID");
    
//-------------------------------------------------

    int Count = 0;
    int nentries = 0;
    while (reader_mc.Next()) {
        nentries++;
    float maxedep = *max_element(hit_edep.begin(), hit_edep.end());
    float minedep = *min_element(hit_edep.begin(), hit_edep.end());
    float thresholdmax = 0.9 * maxedep;
    float thresholdmin = 1.1 * minedep;
        //
    for(int i = 0; i< hit_pdgID.GetSize();++i){
        if (hit_pdgID[i] == 22 && hit_trackID[i] >= 3) {
            for (float edep : hit_edep) {
                if (thresholdmin < edep && edep < thresholdmax) {
                    Count++;
                    }
            }
        }
    }
   
} 
    outfile << "filename" <<filename<<" "<< "Count" << Count <<" "<<"nentries"<<nentries<< endl;
     delete singleFileChain; 
   }
outfile.close();
delete ch_mc;
}
int run(){
    
    readFile();
    cout<<"test 1display"<<endl;
    return 0;
}   
