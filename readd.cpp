#include <iostream>
#include <vector>
#include <string>
using namespace std;
void readd(string path){
    TFile *file = TFile::Open(path);
    TTree *b_events = (TTree*)file->Get("events");
    TBranch *b_fithits = events->GetBranch("fithits");
    b_fithits->SetAddress("trackID",&trackID);

    counts = tree->GetEntries();
    vector<float> trackID;
    int trackID_count = 0;
    for (int i = 0; i < counts; i++){
        b_fithits->GetEntry(i);
        if (trackID == 3&&trackID ==1&&trackID == 2){
            continue;
        }
        else{
            trackID_count++;
            trackID.push_back(trackID);
        }
    }
    cout << trackID_count << endl;
    file->Close();
}
int main (){

    readd("gamma-top10cmx10cmVert-sim.43265645.0.root");
    return 0;
}
