/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package util.data;

/**
 *
 * @author vietan
 */
public class CongressionalTurn extends Turn {

    private char party;
    private char mention;
    private char vote;

    public CongressionalTurn(int index, String speaker, String text) {
        super(index, speaker, text);
    }

    public CongressionalTurn(int index, String speaker, String text,
            char party, char mention, char vote) {
        super(index, speaker, text);
        this.party = party;
        this.mention = mention;
        this.vote = vote;
    }

    public char getMention() {
        return mention;
    }

    public void setMention(char mention) {
        this.mention = mention;
    }

    public char getParty() {
        return party;
    }

    public void setParty(char party) {
        this.party = party;
    }

    public char getVote() {
        return vote;
    }

    public void setVote(char vote) {
        this.vote = vote;
    }
}
